"""gpugen: the synthetic photo generator on the GPU (round 19), made while the net trains (no disk, no CPU workers).

The same paper as flat2/synth.py + geom.py (geom.random_steps: the shapes, isometric bends and folds; geom.random_camera:
the camera), drawn in batches on the GPU:
  - every sample's sheet is a mesh sized so its faces stay ~8 px; all samples' faces are rasterised together
    (z-buffer, perspective-correct barycentrics), so a pixel knows its page coordinates, 3D point and normal;
  - the page image from a pool of pages decoded on the GPU (a coarser copy where the page is drawn small: no aliasing);
  - the paper: its edges torn, serrated, bitten by a spiral binding or a corner torn off (the mask follows them); fine
    relief lit from the light (handled, crumpled paper); the back's print showing through;
  - light: ambient + a direction on the paper's normals; the paper's cast shadow on the table from the light (the same
    mesh flattened along the light: a lifted curl throws its shadow aside), a soft hand / phone shadow, hard-edged
    shadows (a window's edge, blinds), vignetting, glare;
  - the table: real background crops (pagefind2/data.py negcrops), plain / pale / dark / noise / wood surfaces, other
    pages, two surfaces meeting along a line, striped or checked cloth (straight lines that are no page side); things
    lying on it (papers, receipts, cards, objects, a paper stack under the sheet) with contact shadows;
  - over the page (the target stays the whole sheet: the outline goes on behind them): another paper; thumbs, fingers,
    hands and forearms, pens, phones, binder clips, mugs, coins (occluders.py), with their shadows;
  - camera: white balance and mixed light, hand shake, blur, focus falling off across the page, noise growing with the
    light, colour noise, gain, a dim room, the phone's sharpening, JPEG (nvjpeg);
  - framing: normal, small in the frame, cut by the frame, the frame filled by paper; and photos with no page.
Output (page finder v2 layout, pagefind2/net.py): the photo letterboxed into S x S (RGB in [0, 1], padding = the
ImageNet mean), the paper mask, the photo's box, the page's corners, per-side cut flags, the occluders' cover. The
photo is drawn at 2 S and averaged down (antialiased edges, as a phone photo scaled down). SD-HR (sdhr_batch): the
app's crop of the page is chosen from the shape and the camera before anything is drawn, and drawn straight at the
net's input size (each sample's photo px -> canvas px map: ps[i]['A']).

  from gen import GpuGen;  g = GpuGen();  b = g.batch(32)
  python gen.py OUT.jpg [--n 32] [--bench 10]     a contact sheet (mask outline green, corners red) and the speed
"""
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'flat2'))
import geom  # noqa: E402
import synth  # noqa: E402
import occluders  # noqa: E402
import jpeg  # noqa: E402  (GPU JPEG where nvjpeg works, else the CPU)

MEAN = (0.485, 0.456, 0.406)
FRAMING = (('normal', 0.55, (0.55, 1.1)), ('small', 0.15, (0.3, 0.55)), ('cut', 0.22, (1.0, 1.6)), ('fill', 0.08, (1.7, 2.6)))
# SD-HR (the flattening net) sees the app's crop around a page that is in the photo (synth.sample: >= 97 % of the grid inside)
SD_FRAMING = (('normal', 0.85, (0.6, 1.05)), ('small', 0.15, (0.45, 0.6)))
GH, GW = synth.GH, synth.GW
# a page's width in cm by its kind (synth.ASPECTS): the size of a thumb or a pen on it
WIDTH_CM = {'a4': 21.0, 'letter': 21.6, 'receipt': 8.0, 'bill': 15.5, 'card': 8.6, 'coupon_strip': 7.0, 'square': 14.0,
            'landscape': 25.0, 'book': 15.0}


# ---------------------------------------------------------------- helpers

def gauss_blur(x, sigma, max_r=24):
    """x: B x C x H x W, sigma: B (px; < 0.3: unchanged). Separable, a kernel per sample (grouped convolution)."""
    B, C, H, W = x.shape
    if float(sigma.max()) < 0.3:
        return x
    r = int(min(max_r, math.ceil(3 * float(sigma.max()))))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    s = sigma.to(x.dtype).clamp_min(0.3)[:, None]
    k = torch.exp(-0.5 * (t[None] / s) ** 2)
    k = torch.where(sigma[:, None] < 0.3, (t[None] == 0).to(x.dtype), k / k.sum(1, keepdim=True))
    w = k.repeat_interleave(C, 0)[:, None, None, :]                 # (B C) x 1 x 1 x K
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode='replicate'), w, groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode='replicate'), w.transpose(2, 3), groups=B * C)
    return y.reshape(B, C, H, W)


def soft_blur(x, sigma, factor=4):
    """a wide blur (shadows): at 1 / factor of the size, then back up"""
    B, C, H, W = x.shape
    y = F.avg_pool2d(x, factor)
    y = gauss_blur(y, sigma / factor, max_r=48)
    return F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)


def motion_blur(x, lengths, angles):
    """x: B x C x H x W, a straight shake per sample (length px, angle): a line kernel each (grouped convolution)"""
    B, C, H, W = x.shape
    K = int(math.ceil(max(lengths))) | 1
    K = max(K, 3)
    r = K // 2
    ker = torch.zeros(B, K, K, device=x.device)
    t = torch.linspace(-0.5, 0.5, 4 * K, device=x.device)
    for b, (L, th) in enumerate(zip(lengths, angles)):
        px, py = r + t * L * math.cos(th), r + t * L * math.sin(th)
        x0, y0 = px.floor().long(), py.floor().long()
        fx, fy = px - x0, py - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).clamp(0, K - 1), (y0 + dy).clamp(0, K - 1)
            ker[b].index_put_((yi, xi), w, accumulate=True)
    ker = ker / ker.sum((1, 2), keepdim=True)
    w = ker.repeat_interleave(C, 0)[:, None]
    y = F.conv2d(F.pad(x.reshape(1, B * C, H, W), (r, r, r, r), mode='replicate'), w, groups=B * C)
    return y.reshape(B, C, H, W)


def apply_steps_np(P, steps):
    """geom.random_shape's f, from its steps (numpy: the coarse mesh the camera is placed on)"""
    M = P[:, :2].copy()
    for op, args in steps:
        if op == 'bend':
            P = geom.bend(P, *args, M=M)
        elif op == 'fold':
            P = geom.fold(P, *args, M=M)
        else:
            comps, amp = args
            z = np.zeros(len(P))
            for k, th, ph, a in comps:
                z += np.sin(k * (P[:, 0] * math.cos(th) + P[:, 1] * math.sin(th)) + ph) * a
            P = P.copy()
            P[:, 2] += amp * z / 3
    return P


def steps_to_slots(steps, n_slots=8):
    """steps -> (type n_slots: 0 none 1 bend 2 wave, bend n_slots x 6: nx ny c R phi sign (-1: a valley fold),
    wave n_slots x 3 x 4 + amp)"""
    typ = np.zeros(n_slots, np.int64)
    bnd = np.zeros((n_slots, 6), np.float64)
    wav = np.zeros((n_slots, 3, 4), np.float64)
    amp = np.zeros(n_slots, np.float64)
    if len(steps) > n_slots:
        raise ValueError('too many shape steps')
    for i, (op, args) in enumerate(steps):
        if op == 'bend':
            n, c, R, phi = args
            typ[i], bnd[i] = 1, (n[0], n[1], c, R, phi, 1.0)
        elif op == 'fold':
            n, c, angle = args[:3]
            rr = args[3] if len(args) > 3 else 0.004                           # geom.fold's default rounding
            typ[i], bnd[i] = 1, (n[0], n[1], c, max(rr, 1e-4), abs(angle), 1.0 if angle >= 0 else -1.0)
        else:
            comps, a = args
            typ[i], amp[i] = 2, a
            for j, cp in enumerate(comps[:3]):
                wav[i, j] = cp
    return typ, bnd, wav, amp


def deform(P, sid, typ, bnd, wav, amp):
    """the shapes on the GPU: P N x 3 flat sheet points of samples sid; per-sample slots (steps_to_slots, stacked).
    geom.bend's arithmetic, vectorised over every sample's points."""
    M = P[:, :2].clone()
    for k in range(typ.shape[1]):
        t = typ[sid, k]
        if not bool((t > 0).any()):
            continue
        # bend
        b = bnd[sid, k]
        n2, c, R, phi, sg = b[:, :2], b[:, 2], b[:, 3].clamp_min(1e-6), b[:, 4], b[:, 5]
        d = (M * n2).sum(1) - c
        m = (t == 1) & (d > 0)
        qn = (P[:, :2] * n2).sum(1) - c
        off_n, off_z = qn - d, P[:, 2] * sg                 # a valley fold: the mirrored sheet's mountain fold
        a = torch.minimum(d, R * phi) / R
        rest = (d - R * phi).clamp_min(0)
        ca, sa = torch.cos(a), torch.sin(a)
        along = R * sa + ca * (rest + off_n) - sa * off_z
        up = R * (1 - ca) + sa * (rest + off_n) + ca * off_z
        bent = torch.cat([P[:, :2] + (along - qn)[:, None] * n2, (up * sg)[:, None]], 1)
        P = torch.where(m[:, None], bent, P)
        # wave
        w = wav[sid, k]                                   # N x 3 x 4: k, th, ph, a
        z = (torch.sin(w[..., 0] * (P[:, 0:1] * torch.cos(w[..., 1]) + P[:, 1:2] * torch.sin(w[..., 1])) + w[..., 2]) * w[..., 3]).sum(1)
        wz = P[:, 2] + amp[sid, k] * z / 3
        P = torch.where((t == 2)[:, None], torch.cat([P[:, :2], wz[:, None]], 1), P)
    return P


def rasterise(V2, z, Fc, fsid, W, H, n_img, max_box=64):
    """all samples' faces at once. V2 N x 2 canvas px, z N depths, Fc M x 3 vertex ids, fsid M the face's sample,
    canvas W x H, n_img samples -> (face n_img*H*W long, -1 = none; w n_img*H*W x 3 perspective-correct barycentrics).
    geom.rasterise's arithmetic with a sample offset per face (pixel = sample * H * W + y * W + x)."""
    dev = V2.device
    A, B_, C = V2[Fc[:, 0]], V2[Fc[:, 1]], V2[Fc[:, 2]]
    lo = torch.floor(torch.minimum(torch.minimum(A, B_), C)).long()
    hi = torch.ceil(torch.maximum(torch.maximum(A, B_), C)).long()
    span = (hi - lo).clamp(min=0)
    seen = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < W) & (lo[:, 1] < H)
    need = span.max(1).values + 1
    keep = seen & (need <= max_box)
    npix = n_img * H * W
    zbuf = torch.full((npix,), float('inf'), device=dev)
    chunks = []
    for k in (4, 8, 12, 16, 24, 32, 48, 64):
        if k > max_box:
            break
        fi = torch.nonzero(keep & (need <= k)).squeeze(1)
        keep = keep & (need > k)
        if len(fi):
            chunks += [(k, c) for c in torch.split(fi, max(1, 6_000_000 // (k * k)))]
    records = []
    for k, ch in chunks:
        oy, ox = torch.meshgrid(torch.arange(k, device=dev), torch.arange(k, device=dev), indexing='ij')
        px = lo[ch, 0:1] + ox.reshape(1, -1)
        py = lo[ch, 1:2] + oy.reshape(1, -1)
        inb = (px <= hi[ch, 0:1]) & (py <= hi[ch, 1:2]) & (px >= 0) & (py >= 0) & (px < W) & (py < H)
        qx, qy = px.float(), py.float()
        a, b, c = A[ch], B_[ch], C[ch]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
        w0 = ((b[:, 1:2] - c[:, 1:2]) * (qx - c[:, 0:1]) + (c[:, 0:1] - b[:, 0:1]) * (qy - c[:, 1:2])) / den[:, None]
        w1 = ((c[:, 1:2] - a[:, 1:2]) * (qx - c[:, 0:1]) + (a[:, 0:1] - c[:, 0:1]) * (qy - c[:, 1:2])) / den[:, None]
        w2 = 1 - w0 - w1
        inside = inb & (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        f3 = Fc[ch]
        iz0, iz1, iz2 = w0 / z[f3[:, 0:1]], w1 / z[f3[:, 1:2]], w2 / z[f3[:, 2:3]]
        iz = (iz0 + iz1 + iz2).clamp_min(1e-12)
        pix = (fsid[ch, None] * (H * W) + py * W + px)[inside]
        zv = (1 / iz)[inside]
        zbuf.scatter_reduce_(0, pix, zv, reduce='amin')
        records.append((pix, zv, ch[:, None].expand_as(px)[inside],
                        torch.stack([(iz0 / iz)[inside], (iz1 / iz)[inside], (iz2 / iz)[inside]], 1)))
    nf = len(Fc)
    chosen = torch.full((npix,), nf, dtype=torch.long, device=dev)
    for pix, zv, face, _ in records:
        win = zv <= zbuf[pix] + 1e-7
        chosen.scatter_reduce_(0, pix[win], face[win], reduce='amin')
    wts = torch.zeros(npix, 3, device=dev)
    for pix, zv, face, w in records:
        win = face == chosen[pix]
        wts[pix[win]] = w[win]
    chosen[chosen == nf] = -1
    return chosen, wts


class Prof:
    """time per stage (FS_GEN_PROF=1: synchronises the GPU at each mark)"""
    def __init__(self):
        self.on, self.t, self.last = os.environ.get('FS_GEN_PROF') == '1', {}, None

    def __call__(self, name):
        if not self.on:
            return
        torch.cuda.synchronize()
        now = time.time()
        if self.last is not None:
            self.t[self.last[0]] = self.t.get(self.last[0], 0.0) + now - self.last[1]
        self.last = (name, now) if name else None

    def report(self):
        tot = sum(self.t.values()) or 1
        return '  '.join(f'{k} {v / tot * 100:.0f}%' for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- the generator

class GpuGen:
    def __init__(self, pages_dir=os.environ.get('FS_PAGES', 'C:/fs_train/pages2'),
                 bg_pack=os.environ.get('FS_BG_PACK', 'C:/fs_train/pf2/train_neg2.npz'), dev='cuda', pool=128,
                 size=384, seed=0, refresh_every=32, refresh_n=16, neg_p=0.04, stack_p=0.12, face_px=8.0, mode='pf', canvas=0, kinds=None, spread_p=0.08, occ_p=None, sd_size=(1024, 1440)):
        """mode 'pf': page finder samples (batch()), drawn at 2 x size and averaged down; 'sdhr': SD-HR samples
        (sdhr_batch()), the photo drawn at canvas (default 1600) on its long side, as synth.sample's phone photos"""
        self.mode = mode
        # thumbs, fingers, hands, pens, phones, clips, mugs over the page (occluders.py): share of photos with any
        self.occ_p = occ_p if occ_p is not None else (0.15 if mode == 'pf' else 0.10)
        # book spreads: pf sees the whole spread (its gutter target); SD-HR one page of it, as the app cuts a spread
        self.spread_p = spread_p if mode == 'pf' else 0.10
        # paper shapes and their weights (geom.KINDS_R19 / KIND_W_R19); kinds: only these, equally often
        self.kinds = (list(kinds), [1.0] * len(kinds)) if kinds else (geom.KINDS_R19, geom.KIND_W_R19)
        self.dev, self.S = dev, size
        self.C = canvas or (2 * size if mode == 'pf' else 1600)          # the photo's long side
        # the canvas drawn: pf the photo letterboxed (C x C); sdhr the app's crop of the page drawn straight at the net's
        # input size (ps[i]['A']: photo px -> canvas px, set by sdhr_batch), so nothing outside the crop is drawn
        self.CH, self.CW = (self.C, self.C) if mode == 'pf' else (sd_size[1], sd_size[0])
        if mode == 'sdhr':
            neg_p = 0.0
        self.rng = np.random.default_rng(seed)
        self.neg_p, self.stack_p, self.face_px = neg_p, stack_p, face_px
        files = sorted(glob.glob(os.path.join(pages_dir, '*.jpg')))
        if not files:
            raise FileNotFoundError(pages_dir)
        self.page_bytes = [torch.frombuffer(bytearray(open(f, 'rb').read()), dtype=torch.uint8) for f in files]
        self.TH, self.TW = 1088, 840
        self.pool = torch.empty(pool, 3, self.TH, self.TW, dtype=torch.uint8, device=dev)
        self._load_pages(list(range(pool)))
        self.refresh_every, self.refresh_n, self.n_batches = refresh_every, refresh_n, 0
        self.bg = None
        if bg_pack and os.path.exists(bg_pack):
            z = np.load(bg_pack)
            self.bg = (torch.from_numpy(z['jpeg']), z['off'], z['box'])
        self.mean = torch.tensor(MEAN, device=dev)[None, :, None, None]
        self.prof = Prof()

    def _load_pages(self, slots):
        idx = self.rng.integers(0, len(self.page_bytes), len(slots))
        for lo in range(0, len(slots), 16):
            _sync(self.dev)
            ims = jpeg.decode([self.page_bytes[i] for i in idx[lo:lo + 16]], self.dev)
            for s, im in zip(slots[lo:lo + 16], ims):
                self.pool[s] = F.interpolate(im[None].float(), size=(self.TH, self.TW), mode='bilinear', antialias=True,
                                             align_corners=False).round().clamp(0, 255).to(torch.uint8)[0]
        # each page image's brightness (a book's facing pages are picked alike)
        self.pool_mean = (self.pool[:, :, ::16, ::16].float().mean((1, 2, 3)) / 255).cpu().numpy()
        self._ink(slots)

    def _ink(self, slots):
        """self.ink[slots]: where the page has ink (darker than the paper round it), at 1/4 size, spread along lines
        (text lines, rules) a little: the 'foreground' whose flattening matters (ForCenNet, DocGeoNet: text lines)"""
        if not hasattr(self, 'ink'):
            self.ink = torch.zeros(len(self.pool), 1, self.TH // 4, self.TW // 4, dtype=torch.float16, device=self.dev)
        for lo in range(0, len(slots), 16):
            sl = slots[lo:lo + 16]
            g = F.avg_pool2d(self.pool[sl].float().mean(1, keepdim=True) / 255, 2)           # 1/2 size
            bg = F.max_pool2d(F.avg_pool2d(g, 9, 1, 4, count_include_pad=False), 9, 1, 4)    # the paper round it
            ink = ((bg - g) > 0.1).float()
            ink = F.max_pool2d(ink, 2)                                                       # 1/4 size
            ink = F.max_pool2d(ink, (3, 9), 1, (1, 4))                                       # along lines
            self.ink[sl] = ink.half()

    # ------------------------------------------------------------ per-sample choices (CPU, cheap)
    def _sample_params(self, n):
        rng, C = self.rng, self.C
        out = []
        while len(out) < n:
            p = {}
            portrait = rng.random() < 0.75
            ar = [4 / 3, 4 / 3, 4 / 3, 16 / 9, 1.0][int(rng.integers(5))]     # photo shapes: mostly 4:3, some 16:9, square
            lw, sw = C, int(round(C / ar / 2)) * 2
            W, H = (sw, lw) if portrait else (lw, sw)
            p['W'], p['H'] = W, H
            p['ox'], p['oy'] = (C - W) // 2, (C - H) // 2
            p['A'], p['pxs'] = np.array([[1.0, 0.0, p['ox']], [0.0, 1.0, p['oy']]]), 1.0     # photo px -> canvas px, its scale
            r = rng.random()
            acc = 0
            for name, prob, fill in (FRAMING if self.mode == 'pf' else SD_FRAMING):
                acc += prob
                if r < acc:
                    break
            p['framing'] = name
            p['neg'] = rng.random() < self.neg_p
            aname, hw = synth.pick_aspect(rng)
            sx = 1.0
            if rng.random() < self.spread_p:
                # a book spread: two pages (each 1 x hw) on a sheet 2 wide, both rising from a rounded gutter (a book
                # open on the table, spine down), outer parts bent first so the gutter bend carries them
                aname, hw, sx, kind = 'book', rng.uniform(1.3, 1.6), 2.0, 'spread'
                p['half'] = int(rng.integers(2))          # SD-HR: the page it flattens (0 left, 1 right)
                r = rng.uniform(0.02, 0.12)
                steps = [('bend', (np.array([1.0, 0.0]), 1.0, r, rng.uniform(0.15, 0.7))),
                         ('bend', (np.array([-1.0, 0.0]), -1.0, r, rng.uniform(0.15, 0.7)))]
            else:
                kind, steps = geom.random_steps(rng, hw, kinds=self.kinds[0], weights=self.kinds[1])
                if len(steps) > 8:      # (a combo of many creases: the GPU shape holds 8 steps)
                    continue
            P0, _, _ = geom.sheet(hw, 9, 9)
            P0 = P0.reshape(-1, 3) * np.array([sx, 1.0, 1.0])
            Pc = apply_steps_np(P0, steps)
            cam = geom.random_camera(rng, Pc, (W, H), fill=fill, tilt=float(rng.choice([35, 45])), upright=True)
            if name == 'cut':     # off-centre: the frame cuts one or two sides, not all four
                cam.cx += rng.uniform(-0.3, 0.3) * W
                cam.cy += rng.uniform(-0.3, 0.3) * H
            V, zc = cam.project(Pc)
            if (zc <= 0).any():
                continue
            inside = ((V[:, 0] >= 0) & (V[:, 0] < W) & (V[:, 1] >= 0) & (V[:, 1] < H)).mean()
            if inside < 0.25:
                continue
            Vg = V.reshape(9, 9, 2)
            du = np.abs(np.diff(Vg, axis=1)).max(-1).max()
            dv = np.abs(np.diff(Vg, axis=0)).max(-1).max()
            p['nu'] = int(np.clip(math.ceil(8 * du / self.face_px), 16, 220))
            p['nv'] = int(np.clip(math.ceil(8 * dv / self.face_px), 16, 400))
            p.update(kind=kind, aspect=aname, hw=hw, sx=sx, slots=steps_to_slots(steps), cam=cam, tex2=-1)
            # page image: a window of aspect hw in a pool page, sometimes tinted
            th = self.TH / self.TW
            full = rng.random() < 0.5
            if th > hw:
                cw = 1.0 if full else rng.uniform(0.6, 1.0)
                ch = min(1.0, cw * hw / th)
            else:
                ch = 1.0 if full else rng.uniform(0.6, 1.0)
                cw = min(1.0, ch * th / hw)
            p['tex'] = (int(rng.integers(len(self.pool))), rng.uniform(0, 1 - cw), rng.uniform(0, 1 - ch), cw, ch)
            if p['sx'] > 1:   # the facing page: an image of about the same brightness
                near = np.nonzero(np.abs(self.pool_mean - self.pool_mean[p['tex'][0]]) < 0.12)[0]
                p['tex2'] = int(rng.choice(near))
            p['tint'] = rng.uniform(0.85, 1.0, 3) if rng.random() < 0.25 else np.ones(3)
            # light (sheet frame, z up), ambient, shadows, glare
            L = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 1.0])
            p['L'] = L / np.linalg.norm(L)
            p['amb'] = rng.uniform(0.35, 0.75)
            p['cast'] = rng.uniform(0.15, 0.5) if rng.random() < 0.75 else 0.0
            p['cast_sigma'] = rng.uniform(1.5, 6)
            p['hand'] = rng.uniform(0.2, 0.6) if rng.random() < 0.35 else 0.0
            p['hand_blobs'] = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.1, 0.35) * max(W, H), rng.uniform(0.05, 0.2) * max(W, H),
                                rng.uniform(0, math.pi)) for _ in range(int(rng.integers(1, 4)))]
            p['hand_sigma'] = rng.uniform(0.02, 0.1) * max(W, H)
            p['vig'] = rng.uniform(0, 1.0)
            p['glare'] = (rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.05, 0.25) * max(W, H), rng.uniform(20, 90) / 255) if rng.random() < 0.3 else None
            # table
            # (split: two surfaces meeting along a straight line, a table's edge; cloth: stripes or checks. Straight lines
            # beside the page: what an edge finder must not take for the page's side)
            p['bg'] = rng.choice(['real', 'solid', 'wood', 'noise', 'page', 'dark', 'pale', 'split', 'cloth'],
                                 p=[0.26, 0.08, 0.08, 0.12, 0.08, 0.08, 0.18, 0.06, 0.06])
            if p['bg'] == 'real' and self.bg is None:
                p['bg'] = 'pale'
            base = rng.uniform(40, 230, 3)
            if p['bg'] == 'dark':
                base = base * 0.3
            elif p['bg'] == 'pale':
                base = np.full(3, rng.uniform(195, 250)) + rng.uniform(-4, 4, 3)
            p['base'] = base / 255
            p['bg_amp'] = rng.uniform(5, 30) / 255 if p['bg'] == 'noise' else rng.uniform(0, 6) / 255 if p['bg'] == 'pale' else rng.uniform(8, 30) / 255
            p['bg_idx'] = int(rng.integers(len(self.bg[1]) - 1)) if self.bg is not None else 0
            p['bg_page'] = int(rng.integers(len(self.pool)))
            p['bg_flip'] = int(rng.integers(4))
            c2 = rng.uniform(20, 235, 3)
            p['bg_line'] = (rng.uniform(0, math.pi), rng.uniform(0.15, 0.85), rng.uniform(6, 60) * C / 768, rng.random() < 0.5,
                            (c2 if rng.random() < 0.5 else np.full(3, rng.uniform(20, 235))) / 255)   # angle, where, period, checks, colour 2
            # things on the table (under the sheet); a paper stack right under it
            p['clutter'] = [self._clutter(W, H) for _ in range(int(rng.integers(1, 4)))] if rng.random() < 0.55 else []
            p['stack'] = rng.random() < self.stack_p and not p['neg']
            p['edges'] = self._edges(rng) if rng.random() < (0.25 if self.mode == 'pf' else 0.12) else []
            p['over'] = None
            if self.mode == 'pf' and not p['neg'] and rng.random() < 0.08:
                # another paper lying over part of the page: (corner of the page it covers, size, turn, paper image)
                p['over'] = (int(rng.integers(4)), rng.uniform(0.2, 0.55), rng.uniform(-0.6, 0.6), rng.uniform(0.15, 0.45),
                             int(rng.integers(len(self.pool))), rng.uniform(2, 8))
            wcm = WIDTH_CM.get(aname, 21.0) * rng.uniform(0.85, 1.15)
            n_occ = 1 if rng.random() < 0.7 else 2 if rng.random() < 0.85 else 3
            p['occ'] = [occluders.sample(rng, wcm) for _ in range(n_occ)] if rng.random() < (0.05 if p['neg'] else self.occ_p) else []
            p['stack_jit'] = (rng.uniform(-0.04, 0.04, (4, 2)), rng.uniform(-4, 4) * math.pi / 180, rng.uniform(0.85, 1.0))
            # hard-edged shadows (photo px): a window frame's or a person's edge, or blinds' stripes; soft px, strength
            p['hard'] = None
            if rng.random() < 0.15:
                p['hard'] = (rng.random() < 0.4, rng.uniform(0, math.pi), rng.uniform(0.2, 0.8), rng.uniform(0.04, 0.14) * max(W, H),
                             rng.uniform(0.3, 0.7), rng.uniform(0.5, 4.0), rng.uniform(0.2, 0.5))
            # the paper's surface: fine relief lit from the light (crumpled or handled paper), print showing through from
            # the back (thin paper)
            p['relief'] = (rng.uniform(0.04, 0.15) if kind in ('crumple', 'fold2') or kind.startswith('combo') else rng.uniform(0.02, 0.08),
                           int(rng.integers(6, 28))) if rng.random() < 0.3 else None
            p['show'] = (rng.uniform(0.03, 0.12), int(rng.integers(len(self.pool)))) if rng.random() < 0.15 else None
            # camera: white balance (warm / cool, tint) and mixed light (a colour ramp across the photo)
            t_ = rng.uniform(0.08, 0.22) * rng.choice([-1, 1]) if rng.random() < 0.15 else rng.normal(0, 0.04)
            g_ = rng.normal(0, 0.02)
            p['wb'] = np.array([1 + t_, 1 + g_, 1 - t_]) / max(1.0, 1 + abs(t_))
            p['wb2'] = None
            if rng.random() < 0.12:
                t2 = rng.uniform(0.08, 0.2) * rng.choice([-1, 1])
                p['wb2'] = (np.array([1 + t2, 1.0, 1 - t2]) / (1 + abs(t2)), rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.6))
            dim = rng.random() < 0.06                                         # a dim room: low exposure, high ISO noise
            p['gain'] = rng.uniform(0.45, 0.7) if dim else rng.uniform(0.85, 1.15)
            p['noise'] = rng.uniform(3, 8) / 255 if dim else rng.uniform(0.5, 4) / 255
            p['shot'] = rng.uniform(0, 4e-4)                                  # noise growing with the light (variance per unit)
            p['chroma'] = rng.uniform(1, 4) / 255 if rng.random() < 0.3 or dim else 0.0
            p['blur'] = rng.uniform(0.3, 1.6) if rng.random() < 0.5 else 0.0
            # hand shake (length photo px at a 768 photo, angle), focus falling off across the page (sigma, angle, where, width)
            p['motion'] = (rng.uniform(2, 8), rng.uniform(0, math.pi)) if rng.random() < 0.08 else None
            p['dof'] = (rng.uniform(1.5, 4), rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 0.7), rng.uniform(0.15, 0.5)) if rng.random() < 0.1 else None
            p['sharp'] = (rng.uniform(0.2, 1.0), rng.uniform(0.8, 2.0)) if rng.random() < 0.45 else None   # the phone's sharpening
            p['jpeg'] = int(rng.integers(55, 95)) if rng.random() < 0.4 else 0
            out.append(p)
        return out

    def _edges(self, rng):
        """uneven paper edges: [(side 0 top 1 right 2 bottom 3 left, kind, params)] (page units: the sheet's width 1)
        torn: ragged (sums of sines, coarse to fine); teeth: a serrated tear-off edge; bites: a spiral notebook's
        torn holes; corner: a corner torn off along a ragged line"""
        out = []
        sides = [k for k in range(4) if rng.random() < 0.4] or [int(rng.integers(4))]
        for sd in sides:
            kind = str(rng.choice(['torn', 'teeth', 'bites', 'corner'], p=[0.4, 0.3, 0.1, 0.2]))
            if kind == 'torn':
                prm = (rng.uniform(0.004, 0.02), [(rng.uniform(2, 300), rng.uniform(0, 2 * math.pi), rng.uniform(0.2, 1)) for _ in range(7)])
            elif kind == 'teeth':
                prm = (rng.uniform(0.003, 0.01), rng.uniform(0.008, 0.025))
            elif kind == 'bites':
                r = rng.uniform(0.006, 0.012)
                prm = (r, rng.uniform(2.5, 4) * r)
            else:
                prm = (rng.uniform(0.04, 0.15), [(rng.uniform(3, 40), rng.uniform(0, 2 * math.pi), rng.uniform(0.2, 1)) for _ in range(5)])
            out.append((sd, kind, prm))
        return out

    def _edge_keep(self, stp, ps):
        """n x C x C: 1 where the paper is left by its uneven edges (page coordinates stp)"""
        keep = torch.ones(stp.shape[:3], device=stp.device)
        for i, p in enumerate(ps):
            if not p['edges']:
                continue
            s, t = stp[i, ..., 0], stp[i, ..., 1]
            sx, hw = p['sx'], p['hw']
            dist = [t * hw, (1 - s) * sx, (1 - t) * hw, s * sx]       # to the top, right, bottom, left side
            along = [s * sx, t * hw, s * sx, t * hw]
            k = torch.ones_like(s, dtype=torch.bool)
            for sd, kind, prm in p['edges']:
                u = along[sd]
                if kind == 'torn':
                    a, comps = prm
                    f = sum(w * torch.sin(fr * u + ph) for fr, ph, w in comps) / sum(w for _, _, w in comps)
                    k &= dist[sd] >= a * (0.5 + 0.5 * f)
                elif kind == 'teeth':
                    a, per = prm
                    k &= dist[sd] >= a * (2 * torch.frac(u / per) - 1).abs()
                elif kind == 'bites':
                    r, per = prm
                    x = (torch.frac(u / per) - 0.5) * per
                    k &= dist[sd] >= 1.6 * r * torch.sqrt((1 - (x / r) ** 2).clamp_min(0))
                else:   # the corner at the start of side sd (top-left, top-right, bottom-right, bottom-left)
                    L, comps = prm
                    d2 = dist[(sd - 1) % 4]
                    ang = torch.atan2(dist[sd], d2)
                    f = sum(w * torch.sin(fr * ang + ph) for fr, ph, w in comps) / sum(w for _, _, w in comps)
                    k &= dist[sd] + d2 >= L * (1 + 0.15 * f)
            keep[i] = k.float()
        return keep

    def _clutter(self, W, H):
        rng = self.rng
        c = {}
        if rng.random() < 0.65:
            name, hw = synth.pick_aspect(rng)
            c['paper'], c['hw'] = True, hw
            c['tex'] = int(rng.integers(len(self.pool)))
            c['shade'] = rng.uniform(0.75, 1.05)
            L = rng.uniform(0.2, 0.75) * max(W, H)
        else:
            hw = [rng.uniform(1.8, 2.3), rng.uniform(0.5, 1.5), rng.uniform(8, 15)][int(rng.integers(3))]
            c['paper'], c['hw'] = False, hw
            col = np.full(3, rng.uniform(10, 90)) if rng.random() < 0.6 else rng.uniform(30, 230, 3)
            c['col'], c['sheen'] = col / 255, (rng.uniform(-25, 0) / 255, rng.uniform(0, 25) / 255)
            L = rng.uniform(0.1, 0.4) * max(W, H) if hw < 5 else rng.uniform(0.15, 0.3) * max(W, H)
        h = L if hw >= 1 else L * hw
        w = h / hw
        cx, cy, ang = rng.uniform(-0.1, 1.1) * W, rng.uniform(-0.1, 1.1) * H, rng.uniform(-math.pi, math.pi)
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        q = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) * 0.5 * (1 + rng.uniform(-0.06, 0.06, (4, 1)))
        c['quad'] = q @ R.T + [cx, cy]
        c['shadow'] = (rng.uniform(2, 10), rng.uniform(-6, 6), rng.uniform(0, 10), rng.uniform(0.1, 0.4)) if rng.random() < 0.7 else None
        return c

    # ------------------------------------------------------------ drawing
    def _quad_layer(self, quads, texs, cols, sheens, W, H):
        """things lying flat: unit square -> quad (homography) per sample (None: nothing). Returns (the samples drawn (a
        long tensor of indices), their colour k x 3 x H x W, coverage k x 1 x H x W). texs[i]: pool index (a paper) or
        None (cols / sheens: a solid object)"""
        dev = self.dev
        sel = [i for i, q in enumerate(quads) if q is not None]
        quads, texs, cols, sheens = [quads[i] for i in sel], [texs[i] for i in sel], [cols[i] for i in sel], [sheens[i] for i in sel]
        B = len(sel)
        Hinv = np.zeros((B, 3, 3))
        act = np.ones(B, bool)
        for i, q in enumerate(quads):
            src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
            Hinv[i] = np.linalg.inv(_homography(src, np.asarray(q, np.float64)))
        Hi = torch.tensor(Hinv, dtype=torch.float32, device=dev)
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        pts = torch.stack([xx, yy, torch.ones_like(xx)], -1).reshape(1, -1, 3)
        uvw = pts @ Hi.transpose(1, 2)                                   # B x HW x 3
        uv = uvw[..., :2] / uvw[..., 2:].clamp_min(1e-9)
        u, v = uv[..., 0].reshape(B, H, W), uv[..., 1].reshape(B, H, W)
        # coverage with a 1 px soft edge: distance (in px, roughly) inside the unit square
        sc = torch.tensor([max(1.0, np.linalg.norm(np.asarray(q)[1] - np.asarray(q)[0]), np.linalg.norm(np.asarray(q)[3] - np.asarray(q)[0])) if q is not None else 1.0
                           for q in quads], device=dev, dtype=torch.float32)[:, None, None]
        inside = torch.minimum(torch.minimum(u, 1 - u), torch.minimum(v, 1 - v)) * sc
        cov = (inside + 0.5).clamp(0, 1) * torch.tensor(act, device=dev)[:, None, None]
        col = torch.zeros(B, 3, H, W, device=dev)
        tex_i = [i for i in range(B) if act[i] and texs[i] is not None]
        if tex_i:
            t = F.avg_pool2d(self.pool[[texs[i] for i in tex_i]].float() / 255, 2)
            g = torch.stack([u[tex_i] * 2 - 1, v[tex_i] * 2 - 1], -1)
            col[tex_i] = F.grid_sample(t, g, mode='bilinear', padding_mode='border', align_corners=True)
        sol = [i for i in range(B) if texs[i] is None]
        if sol:
            c = torch.tensor(np.stack([cols[i] for i in sol]), device=dev, dtype=torch.float32)[:, :, None, None]
            ab = torch.tensor([sheens[i] for i in sol], device=dev, dtype=torch.float32)
            col[sol] = c + (ab[:, 0, None, None] + (ab[:, 1] - ab[:, 0])[:, None, None] * v[sol].clamp(0, 1))[:, None]
        return torch.tensor(sel, device=dev, dtype=torch.long), col, cov[:, None]

    def render(self, n, ps=None):
        """n photos on the canvas (CH x CW: the photo's pixels through ps[i]['A']) -> state dict (img n x 3 x CH x CW in
        [0, 1], page n x 1 x CH x CW coverage, stp n x CH x CW x 2 page coords, ps, project, shape). ps: the samples'
        choices made beforehand (sdhr_batch: their crops), else drawn here"""
        dev, S, C, CH, CW = self.dev, self.S, self.C, self.CH, self.CW
        self.n_batches += 1
        if self.n_batches % self.refresh_every == 0:
            self._load_pages(list(self.rng.choice(len(self.pool), self.refresh_n, replace=False)))
        self.prof('params')
        ps = self._sample_params(n) if ps is None else ps
        n = len(ps)
        self._last_ps = ps
        self.prof('mesh')
        # ---- meshes, shapes, projection
        Ps, sids, Fs, Fs2, sts, voff = [], [], [], [], [], 0
        for i, p in enumerate(ps):
            nu, nv, hw = p['nu'], p['nv'], p['hw']
            s = torch.linspace(0, 1, nu, device=dev)
            t = torch.linspace(0, 1, nv, device=dev)
            T, Sg = torch.meshgrid(t, s, indexing='ij')
            Ps.append(torch.stack([Sg * p['sx'], T * hw, torch.zeros_like(Sg)], -1).reshape(-1, 3))
            sts.append(torch.stack([Sg, T], -1).reshape(-1, 2))
            sids.append(torch.full((nu * nv,), i, device=dev, dtype=torch.long))
            idx = torch.arange(nu * nv, device=dev).reshape(nv, nu) + voff
            a, b, c, d = idx[:-1, :-1].reshape(-1), idx[:-1, 1:].reshape(-1), idx[1:, 1:].reshape(-1), idx[1:, :-1].reshape(-1)
            Fs.append(torch.cat([torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)]))
            if not p['neg']:
                ri = torch.unique(torch.cat([torch.arange(0, nv, 2, device=dev), torch.tensor([nv - 1], device=dev)]))
                ci = torch.unique(torch.cat([torch.arange(0, nu, 2, device=dev), torch.tensor([nu - 1], device=dev)]))
                i2 = idx[ri][:, ci]
                a, b, c, d = i2[:-1, :-1].reshape(-1), i2[:-1, 1:].reshape(-1), i2[1:, 1:].reshape(-1), i2[1:, :-1].reshape(-1)
                Fs2.append(torch.cat([torch.stack([a, b, c], 1), torch.stack([a, c, d], 1)]))
            voff += nu * nv
        P, sid, Fc, st = torch.cat(Ps), torch.cat(sids), torch.cat(Fs), torch.cat(sts)
        typ, bnd, wav, amp = self._shape(ps)
        P = deform(P, sid, typ, bnd, wav, amp)
        cam = self._cam(ps)
        Am = torch.tensor(np.stack([p['A'] for p in ps]), device=dev, dtype=torch.float32)

        def project(Q, ids):
            q, zz = _project(Q, ids, cam)
            A = Am[ids]
            return torch.stack([A[:, 0, 0] * q[:, 0] + A[:, 0, 1] * q[:, 1] + A[:, 0, 2], A[:, 1, 0] * q[:, 0] + A[:, 1, 1] * q[:, 1] + A[:, 1, 2]], 1), zz
        V2, z = project(P, sid)
        fsid = sid[Fc[:, 0]]
        neg = torch.tensor([p['neg'] for p in ps], device=dev)
        Fpage = Fc[~neg[fsid]]
        self.prof('raster')
        face, wts = rasterise(V2, z, Fpage, fsid[~neg[fsid]], CW, CH, n)
        page = face >= 0
        fv = Fpage[face.clamp_min(0)]                                    # pixel's face's vertices
        stp = (wts[:, :, None] * st[fv]).sum(1)                          # page coords per pixel
        # the paper's back (a flap folded past upright, a dog-ear): the face's winding on screen turns over (page s to
        # the right and t down, as an upright camera sees the front: positive)
        A2, B2, C2 = V2[Fpage[:, 0]], V2[Fpage[:, 1]], V2[Fpage[:, 2]]
        wind = (B2[:, 0] - A2[:, 0]) * (C2[:, 1] - A2[:, 1]) - (B2[:, 1] - A2[:, 1]) * (C2[:, 0] - A2[:, 0])
        back = ((wind < 0)[face.clamp_min(0)] & (face >= 0)).reshape(n, 1, CH, CW).float()
        P3 = P[Fpage]
        nrm = torch.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0], dim=1)
        nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-12)
        nrm = torch.where(nrm[:, 2:3] < 0, -nrm, nrm)
        npx = nrm[face.clamp_min(0)]
        # the paper's cast shadow: the sheet flattened onto the table along the light
        Lt = torch.tensor(np.stack([p['L'] for p in ps]), device=dev, dtype=torch.float32)
        Ls = Lt[sid]
        self.prof('shadow_raster')
        Psh = torch.cat([P[:, :2] - P[:, 2:3] * Ls[:, :2] / Ls[:, 2:3], torch.zeros_like(P[:, 2:3])], 1)
        V2s, zs = project(Psh, sid)
        # (at half size: it is blurred wide)
        F2 = torch.cat(Fs2) if Fs2 else Fpage[:0]
        fsh, _ = rasterise(V2s * 0.5 - 0.25, zs, F2, sid[F2[:, 0]], CW // 2, CH // 2, n)
        shadow = F.interpolate((fsh >= 0).float().reshape(n, 1, CH // 2, CW // 2), size=(CH, CW), mode='bilinear', align_corners=False)
        page = page.reshape(n, 1, CH, CW).float()
        stp = stp.reshape(n, CH, CW, 2)
        if any(p['edges'] for p in ps):
            page = page * self._edge_keep(stp, ps)[:, None]
        npx = npx.reshape(n, CH, CW, 3).permute(0, 3, 1, 2)
        self.prof('texture')
        # ---- the page image: a window of a pool page; a coarser level where it is drawn small
        tex_id = [p['tex'][0] for p in ps]
        win = torch.tensor([p['tex'][1:] for p in ps], device=dev, dtype=torch.float32)  # x0 y0 cw ch (fractions)
        spread = torch.tensor([p['sx'] > 1 for p in ps], device=dev)[:, None, None]
        right = spread & (stp[..., 0] >= 0.5)                                # a spread's right page: its own image
        s_loc = torch.where(spread, torch.where(right, 2 * stp[..., 0] - 1, 2 * stp[..., 0]), stp[..., 0])
        gx = (win[:, 0, None, None] + s_loc * win[:, 2, None, None]) * 2 - 1
        gy = (win[:, 1, None, None] + stp[..., 1] * win[:, 3, None, None]) * 2 - 1
        g = torch.stack([gx, gy], -1)
        tex = self.pool[tex_id].half()
        # texture px per canvas px on the page (its drawn size vs the window's size)
        area = page.sum((1, 2, 3)).clamp_min(1)
        win_px = torch.sqrt(win[:, 2] * self.TW * win[:, 3] * self.TH)
        ratio = (win_px / area.sqrt()).clamp_min(1)
        lvl = torch.log2(ratio).round().clamp(0, 3).long()
        paper = torch.zeros(n, 3, CH, CW, device=dev)
        for L_ in range(4):
            sel = torch.nonzero(lvl == L_).squeeze(1)
            if len(sel):
                t = tex[sel] if L_ == 0 else F.avg_pool2d(tex[sel], 2 ** L_)
                paper[sel] = F.grid_sample(t.float() / 255, g[sel], mode='bilinear', padding_mode='border', align_corners=False)
        del tex
        sp = [i for i, p in enumerate(ps) if p['sx'] > 1]
        if sp:
            t2 = F.avg_pool2d(self.pool[[ps[i]['tex2'] for i in sp]].float() / 255, 2)
            p2 = F.grid_sample(t2, g[sp], mode='bilinear', padding_mode='border', align_corners=False)
            paper[sp] = torch.where(right[sp][:, None], p2, paper[sp])
        # the gutter (the spread's middle line), the page finder's second target
        gut = (spread & ((stp[..., 0] - 0.5).abs() < 0.004) & (page[:, 0] > 0)).float()[:, None]
        sh_i = [i for i, p in enumerate(ps) if p['show'] is not None]
        if sh_i:   # the back's print, mirrored and soft, darkening the front a little
            tb = F.avg_pool2d(self.pool[[ps[i]['show'][1] for i in sh_i]].float() / 255, 4).mean(1, keepdim=True)
            gb = torch.stack([(1 - s_loc[sh_i]) * 2 - 1, stp[sh_i][..., 1] * 2 - 1], -1)
            back_ink = 1 - F.grid_sample(tb, gb, mode='bilinear', padding_mode='border', align_corners=False)
            al = torch.tensor([ps[i]['show'][0] for i in sh_i], device=dev)[:, None, None, None]
            paper[sh_i] = paper[sh_i] * (1 - al * back_ink)
        tint = torch.tensor(np.stack([p['tint'] for p in ps]), device=dev, dtype=torch.float32)[:, :, None, None]
        paper = paper * tint
        # the back: blank paper in the sheet's tint
        tone = tint * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.85, 0.97)
        paper = back * tone + (1 - back) * paper
        self.prof('light')
        # ---- light
        amb = torch.tensor([p['amb'] for p in ps], device=dev)[:, None, None, None]
        lam = (npx * Lt[:, :, None, None]).sum(1, keepdim=True).clamp(0, 1)
        s_paper = amb + (1 - amb) * lam
        s_table = amb + (1 - amb) * Lt[:, 2, None, None, None]
        rl_i = [i for i, p in enumerate(ps) if p['relief'] is not None]
        if rl_i:   # fine relief: a height field on the sheet, its slope lit from the light's direction
            rel = []
            for i in rl_i:
                G = ps[i]['relief'][1]
                h = torch.randn(1, 1, G, G, device=dev)
                h = F.interpolate(h, size=(4 * G, 4 * G), mode='bicubic', align_corners=False)
                h = h.abs() if self.rng.random() < 0.5 else h                   # creased (ridged) or smooth bumps
                rel.append(F.interpolate(h, size=(64, 64), mode='bilinear', align_corners=False) if 4 * G < 64 else F.adaptive_avg_pool2d(h, 64))
            h = torch.cat(rel)
            gs, gt = torch.zeros_like(h), torch.zeros_like(h)
            gs[..., 1:-1] = (h[..., 2:] - h[..., :-2]) / 2
            gt[..., 1:-1, :] = (h[..., 2:, :] - h[..., :-2, :]) / 2
            gq = torch.stack([stp[rl_i][..., 0] * 2 - 1, stp[rl_i][..., 1] * 2 - 1], -1)
            slope = F.grid_sample(torch.cat([gs, gt], 1), gq, mode='bilinear', padding_mode='border', align_corners=False)
            Lr = Lt[rl_i]
            lxy = Lr[:, :2] / Lr[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-6)
            a_ = torch.tensor([ps[i]['relief'][0] for i in rl_i], device=dev)[:, None, None, None]
            s_paper[rl_i] = s_paper[rl_i] * (1 - a_ * (slope[:, 0:1] * lxy[:, 0, None, None, None] + slope[:, 1:2] * lxy[:, 1, None, None, None]))
        yy, xx = torch.meshgrid(torch.arange(CH, device=dev, dtype=torch.float32), torch.arange(CW, device=dev, dtype=torch.float32), indexing='ij')
        # each canvas pixel's photo px (what is placed in the photo: hand shadows, vignetting, glare)
        Ai = torch.tensor(np.stack([np.linalg.inv(np.vstack([p['A'], [0, 0, 1]]))[:2] for p in ps]), device=dev, dtype=torch.float32)[:, :, :, None, None]
        xp = Ai[:, 0, 0] * xx + Ai[:, 0, 1] * yy + Ai[:, 0, 2]
        yp = Ai[:, 1, 0] * xx + Ai[:, 1, 1] * yy + Ai[:, 1, 2]
        pxs = torch.tensor([p['pxs'] for p in ps], device=dev)
        scene = torch.ones(n, 1, CH, CW, device=dev)
        blobs = [(i,) + tuple(bl) for i, p in enumerate(ps) if p['hand'] > 0 for bl in p['hand_blobs']]
        if blobs:
            bt = torch.tensor([b[1:] for b in blobs], device=dev, dtype=torch.float32)[:, :, None, None]
            bi = torch.tensor([b[0] for b in blobs], device=dev, dtype=torch.long)
            X, Y = xp[bi] - bt[:, 0], yp[bi] - bt[:, 1]
            u = X * torch.cos(bt[:, 4]) + Y * torch.sin(bt[:, 4])
            v = -X * torch.sin(bt[:, 4]) + Y * torch.cos(bt[:, 4])
            hand = torch.zeros(n, CH, CW, device=dev).index_put_((bi,), ((u / bt[:, 2]) ** 2 + (v / bt[:, 3]) ** 2 <= 1).float(), accumulate=True).clamp_max(1)
            hs = torch.tensor([p['hand_sigma'] for p in ps], device=dev) * pxs
            hand = soft_blur(hand[:, None], hs) * torch.tensor([p['hand'] for p in ps], device=dev)[:, None, None, None]
            scene = scene * (1 - hand)
        Wt = torch.tensor([p['W'] for p in ps], device=dev, dtype=torch.float32)[:, None, None]
        Ht = torch.tensor([p['H'] for p in ps], device=dev, dtype=torch.float32)[:, None, None]
        vig = torch.tensor([p['vig'] for p in ps], device=dev)[:, None, None, None]
        scene = scene * (1 - vig * (((xp - Wt / 2) / Wt) ** 2 + ((yp - Ht / 2) / Ht) ** 2)[:, None])
        for i, p in enumerate(ps):
            if p['hard'] is not None:
                blinds, th, where, per, duty, soft, amt = p['hard']
                d = xp[i] * math.cos(th) + yp[i] * math.sin(th) - where * max(p['W'], p['H'])
                if blinds:
                    u = torch.remainder(d, per)
                    m = ((torch.minimum(u, per * duty - u.clamp_max(per * duty)) / soft).clamp(0, 1) * (u < per * duty).float())
                else:
                    m = (d / soft).clamp(-0.5, 0.5) + 0.5
                scene[i, 0] = scene[i, 0] * (1 - amt * m)
        # ---- the table and what lies on it
        self.prof('table')
        bg = self._table(ps, xx, yy)
        self.prof('clutter')
        for k in range(3):
            quads = [None] * n
            texs, cols, sheens = [None] * n, [None] * n, [None] * n
            for i, p in enumerate(ps):
                if k < len(p['clutter']):
                    c = p['clutter'][k]
                    quads[i] = c['quad'] @ p['A'][:, :2].T + p['A'][:, 2]
                    if c['paper']:
                        texs[i] = c['tex']
                    else:
                        cols[i], sheens[i] = c['col'], c['sheen']
            if all(q is None for q in quads):
                continue
            sel, col, cov = self._quad_layer(quads, texs, cols, sheens, CW, CH)
            sl = sel.tolist()
            its = [ps[i]['clutter'][k] for i in sl]
            b = bg[sel]
            shs = [c['shadow'] for c in its]
            if any(shs):
                f = [ps[i]['pxs'] for i in sl]
                sh_amt = torch.tensor([sh[3] if sh else 0.0 for sh in shs], device=dev)
                shd = soft_blur(cov, torch.tensor([sh[0] * f_ if sh else 1.0 for sh, f_ in zip(shs, f)], device=dev), factor=2)
                shd = _shift(shd, [sh[1] * f_ if sh else 0.0 for sh, f_ in zip(shs, f)], [sh[2] * f_ if sh else 0.0 for sh, f_ in zip(shs, f)])
                b = b * (1 - sh_amt[:, None, None, None] * shd)
            shade_c = torch.tensor([c.get('shade', 1.0) for c in its], device=dev)
            bg[sel] = col * shade_c[:, None, None, None] * cov + b * (1 - cov)
        # a paper stack: another sheet right under the page (its corners jittered round the page's)
        self.prof('stack')
        stack_q = [None] * n
        for i, p in enumerate(ps):
            if p['stack']:
                c4 = torch.tensor([[0, 0, 0], [1, 0, 0], [1, p['hw'], 0], [0, p['hw'], 0]], device=dev, dtype=torch.float32)
                q, _ = project(c4, torch.full((4,), i, device=dev, dtype=torch.long))
                q = q.cpu().numpy()
                jit, ang, sc = p['stack_jit']
                ctr = q.mean(0)
                Rr = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
                ext = np.ptp(q, 0).max()
                stack_q[i] = (q - ctr) @ Rr.T * (1 + (1 - sc) * 0.2) + ctr + jit * ext
        if any(q is not None for q in stack_q):
            sel, col, cov = self._quad_layer(stack_q, [ps[i]['tex'][0] if stack_q[i] is not None else None for i in range(n)], [None] * n, [None] * n, CW, CH)
            bg[sel] = col * 0.97 * cov + bg[sel] * (1 - cov)
        self.prof('composite')
        cast = torch.tensor([p['cast'] for p in ps], device=dev)[:, None, None, None]
        csig = torch.tensor([p['cast_sigma'] for p in ps], device=dev)
        bg = bg * (1 - cast * soft_blur(shadow, csig, factor=2))
        bg = bg * s_table * scene
        lit = paper * s_paper * scene
        # soft edge: coverage and coverage-weighted paper blurred together (no dark rim)
        one = torch.full((n,), 0.6, device=dev)
        bl = gauss_blur(torch.cat([lit * page, page], 1), one)
        ms = bl[:, 3:]
        img = bl[:, :3] + bg * (1 - ms)
        self.prof('over')
        ov = [i for i, p in enumerate(ps) if p['over'] is not None]
        if ov:
            quads = [None] * n
            for i in ov:
                k, size, ang, wid, _, _ = ps[i]['over']
                hw, sx = ps[i]['hw'], ps[i]['sx']
                cr = torch.tensor([[0, 0, 0], [sx, 0, 0], [sx, hw, 0], [0, hw, 0]], device=dev, dtype=torch.float32)
                q, _ = project(cr, torch.full((4,), i, device=dev, dtype=torch.long))
                q = q.cpu().numpy()
                ext = np.ptp(q, 0).max() * size
                c0 = q[k] + (q.mean(0) - q[k]) * 0.15                    # over the page's corner k
                Rr = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
                quads[i] = (np.array([[-0.5, -wid], [0.5, -wid], [0.5, wid], [-0.5, wid]]) * ext) @ Rr.T + c0
            sel, col, cov = self._quad_layer(quads, [ps[i]['over'][4] if quads[i] is not None else None for i in range(n)], [None] * n, [None] * n, CW, CH)
            k = len(ov)
            shd = _shift(soft_blur(cov, torch.tensor([ps[i]['over'][5] for i in ov], device=dev), factor=2), [4.0] * k, [6.0] * k)
            im = img[sel] * (1 - 0.35 * shd)
            img[sel] = col * s_table[sel] * scene[sel] * cov + im * (1 - cov)
        for i, p in enumerate(ps):
            if p['glare'] is not None:
                gx_, gy_, r, a = p['glare']
                blob = torch.exp(-((xp[i] - gx_) ** 2 + (yp[i] - gy_) ** 2) / (2 * r * r)) * a
                img[i] = img[i] + blob[None] * ms[i]
        self.prof('occluders')
        occ = self._occlude(img, ps, project, (typ, bnd, wav, amp), s_table, scene, ms) if any(p['occ'] for p in ps) else None
        self.prof('camera')
        # ---- camera
        gain = torch.tensor([p['gain'] for p in ps], device=dev)[:, None, None, None]
        sig_n = torch.tensor([p['noise'] for p in ps], device=dev)[:, None, None, None]
        img = img * torch.tensor(np.stack([p['wb'] for p in ps]), device=dev, dtype=torch.float32)[:, :, None, None]
        for i, p in enumerate(ps):
            if p['wb2'] is not None:   # mixed light: the other light's colour taking over across the photo
                wb2, th, where, width = p['wb2']
                d = (xp[i] / p['W'] - 0.5) * math.cos(th) + (yp[i] / p['H'] - 0.5) * math.sin(th) + 0.5 - where
                ramp = (d / width + 0.5).clamp(0, 1)[None]
                img[i] = img[i] * (1 - ramp + ramp * torch.tensor(wb2 / p['wb'], device=dev, dtype=torch.float32)[:, None, None])
        k_ = [p['pxs'] * self.C / 768 for p in ps]                            # photo px at 768 -> canvas px
        mb = [i for i, p in enumerate(ps) if p['motion'] is not None]
        if mb:
            img[mb] = motion_blur(img[mb], [ps[i]['motion'][0] * k_[i] for i in mb], [ps[i]['motion'][1] for i in mb])
        img = img * gain
        shot = torch.tensor([p['shot'] for p in ps], device=dev)[:, None, None, None]
        img = img + torch.randn(img.shape, device=dev, dtype=torch.float16) * torch.sqrt(sig_n ** 2 + shot * img.clamp(0, 1))
        ci = [i for i, p in enumerate(ps) if p['chroma'] > 0]
        if ci:   # colour noise, blotchy (a phone's denoiser leaves it at a coarser scale)
            cn = torch.randn(len(ci), 3, CH // 4 + 1, CW // 4 + 1, device=dev)
            img[ci] = img[ci] + F.interpolate(cn, size=(CH, CW), mode='bilinear', align_corners=False) * torch.tensor(
                [ps[i]['chroma'] for i in ci], device=dev)[:, None, None, None]
        img = img.clamp(0, 1)
        bi = [i for i, p in enumerate(ps) if p['blur'] >= 0.3]
        if bi:
            img[bi] = gauss_blur(img[bi], torch.tensor([ps[i]['blur'] * ps[i]['pxs'] for i in bi], device=dev))
        di = [i for i, p in enumerate(ps) if p['dof'] is not None]
        if di:   # out of focus towards one side (a page photographed at a slant, close)
            bl = gauss_blur(img[di], torch.tensor([ps[i]['dof'][0] * k_[i] for i in di], device=dev))
            for j, i in enumerate(di):
                sg, th, where, width = ps[i]['dof']
                d = (xp[i] / ps[i]['W'] - 0.5) * math.cos(th) + (yp[i] / ps[i]['H'] - 0.5) * math.sin(th) + 0.5 - where
                ramp = (d / width).clamp(0, 1)[None]
                img[i] = img[i] * (1 - ramp) + bl[j] * ramp
        si_ = [i for i, p in enumerate(ps) if p['sharp'] is not None]
        if si_:   # the phone's sharpening (halos at edges)
            sm = gauss_blur(img[si_], torch.tensor([ps[i]['sharp'][1] * ps[i]['pxs'] for i in si_], device=dev))
            am = torch.tensor([ps[i]['sharp'][0] for i in si_], device=dev)[:, None, None, None]
            img[si_] = (img[si_] + am * (img[si_] - sm)).clamp(0, 1)
        q = [p['jpeg'] for p in ps]
        self.prof('jpeg')
        ji = [i for i in range(n) if q[i]]
        if ji:
            u8 = (img[ji] * 255).round().clamp(0, 255).to(torch.uint8)
            enc = jpeg.encode_many([u8[j] for j in range(len(ji))], [q[i] for i in ji])
            _sync(dev)
            dec = jpeg.decode(enc, dev)   # nvjpeg encodes on the GPU, decodes from host bytes (or the CPU: jpeg.py)
            img[ji] = torch.stack(dec).float() / 255
        self.prof('after_render')
        return dict(img=img.clamp(0, 1), page=page, stp=stp, ps=ps, project=project, shape=(typ, bnd, wav, amp), neg=neg, gut=gut, occ=occ)

    def _occlude(self, img, ps, project, shape, s_table, scene, ms):
        """draws the occluders (occluders.py) into img in place, each round its anchor on a side of the page with the
        page's projection taken as affine there. -> coverage n x 1 x C x C"""
        dev, CH, CW = self.dev, self.CH, self.CW
        typ, bnd, wav, amp = shape
        occ = torch.zeros(len(ps), 1, CH, CW, device=dev)
        jobs = [(i, o) for i, p in enumerate(ps) for o in p['occ']]
        e = 0.01
        Q, ids = [], []
        for i, o in jobs:
            sx, hw, u = ps[i]['sx'], ps[i]['hw'], o['u']
            a = [(u * sx, 0.0), (sx, u * hw), (u * sx, hw), (0.0, u * hw)][o['side']]
            Q += [(a[0], a[1], 0.0), (a[0] + e, a[1], 0.0), (a[0], a[1] + e, 0.0)]
            ids += [i] * 3
        ids = torch.tensor(ids, device=dev, dtype=torch.long)
        q2, _ = project(deform(torch.tensor(Q, device=dev, dtype=torch.float32), ids, typ, bnd, wav, amp), ids)
        q2 = q2.reshape(-1, 3, 2).cpu().numpy().astype(np.float64)
        for k, (i, o) in enumerate(jobs):
            ap = q2[k, 0]
            J = np.stack([(q2[k, 1] - ap) / e, (q2[k, 2] - ap) / e], 1)          # px per page unit
            inw = np.array([(0, 1), (-1, 0), (0, -1), (1, 0)][o['side']], np.float64)
            ca, sa = math.cos(o['ang']), math.sin(o['ang'])
            din = np.array([inw[0] * ca - inw[1] * sa, inw[0] * sa + inw[1] * ca])
            Rl = np.stack([din, [-din[1], din[0]]])                              # page frame -> local (U into the page, V)
            M = J @ Rl.T / o['wcm']                                              # px per local cm
            if not np.isfinite(M).all() or abs(np.linalg.det(M)) < 1e-3:
                continue
            L = ps[i]['L']
            sl = Rl @ (-o['h'] * L[:2] / L[2])                                  # its shadow's offset (local cm)
            lu, lv = Rl @ (L[:2] / max(1e-6, np.linalg.norm(L[:2])))
            sig = o['blur'] * self.C / 768 * ps[i]['pxs']
            pts = occluders.extent(o)
            pp = ap + np.concatenate([pts, pts + sl]) @ M.T
            pad = 3 + 3 * sig
            x0, y0 = np.maximum(np.floor(pp.min(0) - pad), 0).astype(int)
            x1, y1 = np.minimum(np.ceil(pp.max(0) + pad), [CW, CH]).astype(int)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            Mi = np.linalg.inv(M)
            yy, xx = torch.meshgrid(torch.arange(y0, y1, device=dev, dtype=torch.float32) - ap[1],
                                    torch.arange(x0, x1, device=dev, dtype=torch.float32) - ap[0], indexing='ij')
            U, V = Mi[0, 0] * xx + Mi[0, 1] * yy, Mi[1, 0] * xx + Mi[1, 1] * yy
            col, cov, sh, und = occluders.draw(o, U, V, U - sl[0], V - sl[1], math.sqrt(abs(np.linalg.det(Mi))), lu, lv)
            cov = cov * (1 - und * ms[i, 0, y0:y1, x0:x1])                     # a hand behind the page: only past it
            reg = img[i, :, y0:y1, x0:x1] * (1 - o['dark'] * sh[None])
            lit = col * s_table[i, 0] * scene[i, :, y0:y1, x0:x1] * cov[None]
            if sig > 0.3:
                b = gauss_blur(torch.cat([lit, cov[None]])[None], torch.tensor([sig], device=dev))[0]
                lit, cov = b[:3], b[3]
            img[i, :, y0:y1, x0:x1] = reg * (1 - cov[None]) + lit
            occ[i, 0, y0:y1, x0:x1] = torch.maximum(occ[i, 0, y0:y1, x0:x1], cov)
        return occ

    def batch(self, n):
        """page finder: n samples -> dict(x n x 3 x S x S, m n x 1 x S x S, v n x 1 x S x S, box n x 4, corners n x 4 x 2,
        cut n x 4 (side of the frame cutting the page: top, right, bottom, left of the page), kind, framing)"""
        dev, S, C = self.dev, self.S, self.C
        r = self.render(n)
        img, page, ps, project, neg = r['img'], r['page'], r['ps'], r['project'], r['neg']
        typ, bnd, wav, amp = r['shape']
        # ---- down to S, letterbox, targets
        x = F.avg_pool2d(img, 2)
        m = (F.avg_pool2d(page, 2) > 0.5).float()
        box = torch.tensor([[p['ox'] // 2, p['oy'] // 2, p['ox'] // 2 + p['W'] // 2, p['oy'] // 2 + p['H'] // 2] for p in ps], dtype=torch.long)
        bd = box.to(dev)
        ar = torch.arange(S, device=dev)
        v = (((ar[None, :] >= bd[:, 0:1]) & (ar[None, :] < bd[:, 2:3]))[:, None, None, :]
             & ((ar[None, :] >= bd[:, 1:2]) & (ar[None, :] < bd[:, 3:4]))[:, None, :, None]).float()
        x = x * v + (1 - v) * self.mean
        m = m * v
        occ = F.avg_pool2d(r['occ'], 2) * v if r['occ'] is not None else torch.zeros_like(m)
        # the corners as drawn (through the same shape: a lifted corner moves) and 33 points along each side, every
        # sample's together
        tt = torch.linspace(0, 1, 33, device=dev)
        st = torch.cat([torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], device=dev, dtype=torch.float32),
                        torch.stack([tt, 0 * tt], 1), torch.stack([1 + 0 * tt, tt], 1), torch.stack([tt, 1 + 0 * tt], 1), torch.stack([0 * tt, tt], 1)])
        K = len(st)
        sxhw = torch.tensor([[p['sx'], p['hw']] for p in ps], device=dev, dtype=torch.float32)
        Q = torch.cat([(st[None] * sxhw[:, None]).reshape(-1, 2), torch.zeros(n * K, 1, device=dev)], 1)
        ids = torch.arange(n, device=dev).repeat_interleave(K)
        q2, _ = project(deform(Q, ids, typ, bnd, wav, amp), ids)
        q2 = q2.reshape(n, K, 2)
        corners = (q2[:, :4] - 0.5) / 2                     # canvas px -> S px (2 x 2 average)
        corners[neg] = float('nan')                         # a photo with no page: no corners
        # sides cut by the frame: the side's points outside the photo (> 30 %)
        o = torch.tensor([[p['ox'], p['oy'], p['W'], p['H']] for p in ps], device=dev, dtype=torch.float32)[:, None, None]
        sd = q2[:, 4:].reshape(n, 4, 33, 2)
        qx, qy = sd[..., 0] - o[..., 0], sd[..., 1] - o[..., 1]
        out = (qx < 0) | (qy < 0) | (qx > o[..., 2] - 1) | (qy > o[..., 3] - 1)
        cut = ((out.float().mean(-1) > 0.3) & ~neg[:, None]).cpu()
        gut = (F.avg_pool2d(r['gut'], 2) > 0.2).float() * v                 # the gutter line, >= 1 px wide at S
        self.prof(None)
        return dict(x=x.clamp(0, 1), m=m, v=v, box=box.to(dev), corners=corners, cut=cut, gut=gut, occ=occ,
                    kind=[p['kind'] for p in ps], framing=[p['framing'] if not p['neg'] else 'none' for p in ps])

    KINDS = ['flat', 'nearflat', 'curl', 'corner', 'fold', 'fold2', 'curl_corner', 'wave_curl']   # train_sdhr.Mixed.KINDS
    AS_SD = {'fold_sharp': 'fold', 'crumple': 'fold2', 'dogear': 'corner', 'spread': 'curl'}   # round-19 kinds, as SD-HR logs them

    def sdhr_batch(self, n, margin=(0.02, 0.10), jitter=0.02):
        """SD-HR: n samples as synth.sample + synth.app_input make them -> dict(x n x 3 x H x W uint8 RGB (the app's crop,
        turned when the page lies wider than tall), t n x 2 x GH x GW float32 (the backward map in [-1, 1] of the crop),
        kind n long (KINDS), z n x GH x GW (the sheet's height at each grid point, page widths), turned n bool,
        fg n x GH x GW uint8 (0-255: ink on the page at each target grid point: text lines, rules, graphics)).
        A book spread gives one of its pages (ps[i]['half']), the facing page beyond the gutter in the crop's margin.
        Left out as synth.sample does: < 97 % of the grid in the photo, or the paper hiding > 1 % of itself; and an
        occluder (hand, pen, phone) over > 8 % of the grid.
        The crop is chosen before anything is drawn (the grid needs only the shape and the camera), and drawn straight at
        the net's input size: canvas px = photo px through the crop, its turn and the resize (pixel centres as cv2 and
        F.interpolate place them), so no pixel outside the crop is drawn and no photo is drawn to be thrown away."""
        assert self.mode == 'sdhr'
        dev, CW, CH = self.dev, self.CW, self.CH
        K = GH * GW
        ti, sj = torch.meshgrid(torch.linspace(0, 1, GH, device=dev), torch.linspace(0, 1, GW, device=dev), indexing='ij')
        si, tj = torch.meshgrid(torch.linspace(0, 1, GH, device=dev), 1 - torch.linspace(0, 1, GW, device=dev), indexing='ij')
        want = torch.stack([sj.reshape(-1), ti.reshape(-1)], 1)
        interior = torch.zeros(GH, GW, dtype=torch.bool, device=dev)
        interior[2:-2, 2:-2] = True
        interior = interior.reshape(-1)
        xs, ts, ks, zs, tus, fg, meta = [], [], [], [], [], [], []
        while len(xs) < n:
            self.prof('sd_plan')
            ps = self._sample_params(int(math.ceil((n - len(xs)) * 1.1)))
            m = len(ps)
            cam, shape = self._cam(ps), self._shape(ps)
            # a page's x on the sheet: s (one page), or half + s (the page of a spread: each page 1 wide)
            off = torch.tensor([float(p.get('half', 0)) if p['sx'] > 1 else 0.0 for p in ps], device=dev)
            sxhw = torch.tensor([[p['sx'], p['hw']] for p in ps], device=dev, dtype=torch.float32)

            def grid_px(sv, tv, sel):
                """the page's grid (page s, t) of samples sel -> photo px len(sel) x K x 2, its height len(sel) x K"""
                ids = torch.tensor(sel, device=dev, dtype=torch.long).repeat_interleave(K)
                Q = torch.stack([sv.reshape(-1).repeat(len(sel)) + off[ids], tv.reshape(-1).repeat(len(sel)) * sxhw[ids, 1],
                                 torch.zeros(len(ids), device=dev)], 1)
                Qd = deform(Q, ids, *shape)
                q2, _ = _project(Qd, ids, cam)
                return q2.reshape(len(sel), K, 2), Qd[:, 2].reshape(len(sel), K)
            g2, gz = grid_px(sj, ti, list(range(m)))
            WH = torch.tensor([[p['W'], p['H']] for p in ps], device=dev, dtype=torch.float32)[:, None]
            gr = g2.round()
            inside = ((gr >= 0) & (gr < WH)).all(-1).float().mean(1).cpu().numpy()
            corners = g2[:, [0, GW - 1, K - 1, K - GW]].cpu().numpy().astype(np.float64)   # grid rows / cols (0, 0) (0, -1) (-1, -1) (-1, 0)
            keep, crops = [], {}
            for i, p in enumerate(ps):
                if inside[i] < 0.97:
                    continue
                c = corners[i]
                ext = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
                c = c + self.rng.uniform(-jitter, jitter, c.shape) * ext
                x0, y0 = c.min(0)
                x1, y1 = c.max(0)
                mg = self.rng.uniform(*margin) * max(x1 - x0, y1 - y0)
                bx0, by0 = max(0, int(math.floor(x0 - mg))), max(0, int(math.floor(y0 - mg)))
                bx1, by1 = min(p['W'], int(math.ceil(x1 + mg)) + 1), min(p['H'], int(math.ceil(y1 + mg)) + 1)
                if bx1 - bx0 < 16 or by1 - by0 < 16:
                    continue
                turned = bool(np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[3] - c[0]))
                cw, ch = bx1 - bx0, by1 - by0
                if turned:   # the crop turned clockwise (cw x ch -> ch x cw: column ch - 1 - y, row x), then resized
                    kx, ky = CW / ch, CH / cw
                    A = np.array([[0.0, -kx, kx * (ch - 0.5 + by0) - 0.5], [ky, 0.0, ky * (0.5 - bx0) - 0.5]])
                else:
                    kx, ky = CW / cw, CH / ch
                    A = np.array([[kx, 0.0, kx * (0.5 - bx0) - 0.5], [0.0, ky, ky * (0.5 - by0) - 0.5]])
                p['A'], p['pxs'] = A, math.sqrt(kx * ky)
                k = max(kx, ky)                                                  # the mesh's faces stay ~8 canvas px
                p['nu'] = int(np.clip(math.ceil(p['nu'] * k), 16, 260))
                p['nv'] = int(np.clip(math.ceil(p['nv'] * k), 16, 460))
                keep.append(i)
                crops[i] = (bx0, by0, cw, ch, turned)
            if not keep:
                continue
            self.prof('render')
            r = self.render(len(keep), ps=[ps[i] for i in keep])
            self.prof('sd_targets')
            img, stp = r['img'], r['stp']
            # the paper hiding itself: the drawn page coordinates at the grid's points
            Am = torch.tensor(np.stack([ps[i]['A'] for i in keep]), device=dev, dtype=torch.float32)[:, None]
            gk = g2[keep]
            gx = (Am[..., 0, 0] * gk[..., 0] + Am[..., 0, 1] * gk[..., 1] + Am[..., 0, 2]).round().long()
            gy = (Am[..., 1, 0] * gk[..., 0] + Am[..., 1, 1] * gk[..., 1] + Am[..., 1, 2]).round().long()
            inc = (gx >= 0) & (gy >= 0) & (gx < CW) & (gy < CH) & interior[None]
            fgs = self._grid_ink([ps[i] for i in keep], [crops[i][4] for i in keep], si, tj, sj, ti)
            for j, i in enumerate(keep):
                if len(xs) >= n:
                    break
                sl = inc[j]
                got = stp[j, gy[j, sl], gx[j, sl]]
                # (stp: the sheet's s, 0-1 over both pages of a spread)
                w = torch.stack([(want[:, 0] + off[i]) / sxhw[i, 0], want[:, 1]], 1)
                bad = ~torch.isfinite(got[:, 0]) | ((got - w[sl]).abs().max(1).values > 0.02)
                if bad.float().mean() > 0.01:
                    continue
                # a hand / pen / phone over the page: some is real (fingers holding it), most of it hidden is not a
                # page SD-HR can flatten (the round-19 audit: a hand over a whole crop). Out above 8 % of the grid
                if r['occ'] is not None and (r['occ'][j, 0, gy[j, sl], gx[j, sl]] > 0.5).float().mean() > 0.08:
                    continue
                bx0, by0, cw, ch, turned = crops[i]
                if turned:   # net row i: page s = i / (GH - 1); net column j: page t = 1 - j / (GW - 1)
                    grid = grid_px(si, tj, [i])[0][0].T.reshape(2, GH, GW)
                else:
                    grid = g2[i].T.reshape(2, GH, GW)
                t = grid - torch.tensor([bx0, by0], device=dev, dtype=torch.float32)[:, None, None]
                if turned:
                    t = torch.stack([(ch - 1) - t[1], t[0]])
                    cw, ch = ch, cw
                xs.append((img[j] * 255).round().clamp(0, 255).to(torch.uint8))
                ts.append(torch.stack([t[0] / (cw - 1) * 2 - 1, t[1] / (ch - 1) * 2 - 1]))
                k = ps[i]['kind']
                ks.append(self.KINDS.index(self.AS_SD.get(k, 'fold2' if k.startswith('combo') else k)))
                zs.append(gz[i].reshape(GH, GW))
                tus.append(turned)
                fg.append(fgs[j])
                meta.append(dict(kind=k, sx=ps[i]['sx'], half=ps[i].get('half'), stack=ps[i]['stack'], tex=ps[i]['tex'], tex2=ps[i].get('tex2'), turned=turned))
        self.prof(None)
        return dict(x=torch.stack(xs), t=torch.stack(ts).float(), kind=torch.tensor(ks, device=dev), z=torch.stack(zs),
                    turned=torch.tensor(tus), fg=torch.stack(fg), meta=meta)

    def _grid_ink(self, ps, turned, si, tj, sj, ti):
        """the ink (self.ink) at each sample's target grid -> len(ps) x GH x GW uint8. The target's grid point (row i,
        column j) is page (s, t) = (sj, ti) upright, (si, tj) turned; the page image there: its window of the pool page
        (a spread's right page: the facing image tex2, the same window)"""
        dev = self.dev
        s = torch.stack([si if tu else sj for tu in turned])                      # n x GH x GW
        t = torch.stack([tj if tu else ti for tu in turned])
        win = torch.tensor([p['tex'][1:] for p in ps], device=dev, dtype=torch.float32)[:, :, None, None]
        idx = [p['tex2'] if p['sx'] > 1 and p.get('half', 0) == 1 else p['tex'][0] for p in ps]
        g = torch.stack([(win[:, 0] + s * win[:, 2]) * 2 - 1, (win[:, 1] + t * win[:, 3]) * 2 - 1], -1)
        v = F.grid_sample(self.ink[idx].float(), g, mode='bilinear', padding_mode='border', align_corners=False)[:, 0]
        return (v * 255).round().to(torch.uint8)

    def _cam(self, ps):
        dev = self.dev
        return (torch.tensor(np.stack([p['cam'].R for p in ps]), device=dev, dtype=torch.float32),
                torch.tensor(np.stack([p['cam'].t for p in ps]), device=dev, dtype=torch.float32),
                torch.tensor([[p['cam'].f, p['cam'].cx, p['cam'].cy] for p in ps], device=dev, dtype=torch.float32))

    def _shape(self, ps):
        dev = self.dev
        return (torch.tensor(np.stack([p['slots'][0] for p in ps]), device=dev),
                torch.tensor(np.stack([p['slots'][1] for p in ps]), device=dev, dtype=torch.float32),
                torch.tensor(np.stack([p['slots'][2] for p in ps]), device=dev, dtype=torch.float32),
                torch.tensor(np.stack([p['slots'][3] for p in ps]), device=dev, dtype=torch.float32))

    def _table(self, ps, xx, yy):
        dev, CH, CW, n = self.dev, self.CH, self.CW, len(ps)
        base = torch.tensor(np.stack([p['base'] for p in ps]), device=dev, dtype=torch.float32)[:, :, None, None]
        bg = base.expand(n, 3, CH, CW).clone()
        amp = torch.tensor([p['bg_amp'] for p in ps], device=dev)[:, None, None, None]
        low = F.interpolate(torch.randn(n, 1, CH // 16 + 1, CW // 16 + 1, device=dev), size=(CH, CW), mode='bicubic', align_corners=False)
        kinds = [p['bg'] for p in ps]
        nz = torch.tensor([k in ('noise', 'pale') for k in kinds], device=dev)[:, None, None, None]
        bg = bg + nz * low * amp
        for i, k in enumerate(kinds):
            if k == 'wood':
                r = self.rng
                y = torch.linspace(0, r.uniform(20, 80), CH, device=dev)[:, None] + torch.randn(CH, 1, device=dev).cumsum(0) * 0.05
                gw = torch.sin(y + torch.linspace(0, r.uniform(1, 6), CW, device=dev)[None]) * amp[i, 0, 0, 0]
                bg[i] = bg[i] + gw[None]
            if k in ('split', 'cloth'):
                th, where, per, checks, c2 = ps[i]['bg_line']
                c2 = torch.tensor(c2, device=dev, dtype=torch.float32)[:, None, None]
                d = (xx - CW / 2) * math.cos(th) + (yy - CH / 2) * math.sin(th)
                if k == 'split':   # two surfaces meeting along a line through the photo
                    m = (d / 1.0 - (where - 0.5) * max(CW, CH)).clamp(-0.5, 0.5) + 0.5
                    bg[i] = bg[i] * (1 - m) + (c2 + low[i] * amp[i, 0]) * m
                else:              # stripes, or checks (two crossing stripe sets)
                    m = (torch.sin(d * 2 * math.pi / per) * per / 4).clamp(-0.5, 0.5) + 0.5
                    if checks:
                        d2 = -(xx - CW / 2) * math.sin(th) + (yy - CH / 2) * math.cos(th)
                        m2 = (torch.sin(d2 * 2 * math.pi / per) * per / 4).clamp(-0.5, 0.5) + 0.5
                        m = (m - m2).abs()
                    bg[i] = bg[i] * (1 - m) + c2 * m
        pg = [i for i, k in enumerate(kinds) if k == 'page']
        if pg:
            t = self.pool[[ps[i]['bg_page'] for i in pg]].float() / 255
            bg[pg] = F.interpolate(t, size=(CH, CW), mode='bilinear', antialias=True, align_corners=False) * torch.tensor(
                [self.rng.uniform(0.5, 0.9) for _ in pg], device=dev)[:, None, None, None]
        rl = [i for i, k in enumerate(kinds) if k == 'real']
        if rl:
            jb, off, bx = self.bg
            _sync(dev)
            ims = jpeg.decode([jb[off[ps[i]['bg_idx']]:off[ps[i]['bg_idx'] + 1]] for i in rl], dev)
            for i, im in zip(rl, ims):
                x0, y0, x1, y1 = [int(v) for v in bx[ps[i]['bg_idx']]]
                crop = torch.rot90(im[:, y0:y1, x0:x1], ps[i]['bg_flip'], (1, 2)).float()[None] / 255
                bg[i] = F.interpolate(crop, size=(CH, CW), mode='bilinear', align_corners=False)[0]
        bg = bg + torch.randn_like(bg) * torch.tensor([self.rng.uniform(1, 6) / 255 for _ in ps], device=dev)[:, None, None, None]
        return bg.clamp(0, 1)


def _project(Q, ids, cam):
    """sheet points Q (N x 3) of samples ids -> photo px N x 2, depth N (cam: GpuGen._cam)"""
    Rm, tv, fcx = cam
    X = torch.einsum('nij,nj->ni', Rm[ids], Q * torch.tensor([1.0, 1.0, -1.0], device=Q.device)) + tv[ids]
    zz = X[:, 2]
    return torch.stack([fcx[ids, 0] * X[:, 0] / zz + fcx[ids, 1], fcx[ids, 0] * X[:, 1] / zz + fcx[ids, 2]], 1), zz


def _sync(dev):
    """before a GPU JPEG decode: nvjpeg runs on its own stream (SPEED.md: without this the trainer's weights went NaN)"""
    if str(dev).startswith('cuda'):
        torch.cuda.synchronize()


def _homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, Vt = np.linalg.svd(np.array(A))
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def _shift(x, dx, dy):
    """per-sample translation by (dx, dy) px (bilinear)"""
    B, _, H, W = x.shape
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = 1
    th[:, 1, 1] = 1
    th[:, 0, 2] = -2 * torch.tensor(dx, device=x.device, dtype=torch.float32) / W
    th[:, 1, 2] = -2 * torch.tensor(dy, device=x.device, dtype=torch.float32) / H
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


if __name__ == '__main__':
    import argparse
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--n', type=int, default=24); ap.add_argument('--bench', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--pool', type=int, default=128); ap.add_argument('--dev', default='cuda'); ap.add_argument('--pages', default=''); ap.add_argument('--kinds', default=''); ap.add_argument('--spread_p', type=float, default=0.08)
    a = ap.parse_args()
    t0 = time.time()
    g = GpuGen(seed=a.seed, pool=a.pool, dev=a.dev, kinds=[k for k in a.kinds.split(',') if k] or None, spread_p=a.spread_p, **({'pages_dir': a.pages} if a.pages else {}))
    sync = torch.cuda.synchronize if a.dev == 'cuda' else (lambda: None)
    sync()
    print(f'init {time.time() - t0:.1f} s', flush=True)
    b = g.batch(a.n)
    if a.bench:
        sync()
        t0 = time.time()
        for _ in range(a.bench):
            b = g.batch(a.n)
        sync()
        dt = time.time() - t0
        print(f'{a.bench} batches of {a.n}: {a.bench * a.n / dt:.1f} photos/s, peak GPU {torch.cuda.max_memory_allocated() / 1e9 if a.dev == "cuda" else 0:.2f} GB', flush=True)
    tiles = []
    for i in range(a.n):
        im = (b['x'][i].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
        cs, _ = cv2.findContours(b['m'][i, 0].cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (40, 220, 40), 1)
        co, _ = cv2.findContours((b['occ'][i, 0].cpu().numpy() > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, co, -1, (230, 60, 230), 1)
        im[b['gut'][i, 0].cpu().numpy() > 0] = (0, 230, 255)
        for (cx, cy) in b['corners'][i].cpu().numpy():
            if not np.isfinite(cx):   # a photo with no page
                continue
            cv2.circle(im, (int(round(cx)), int(round(cy))), 3, (40, 40, 230), -1)
        cv2.putText(im, f"{b['kind'][i]} {b['framing'][i]} cut{''.join('TRBL'[k] for k in range(4) if b['cut'][i, k])}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(a.out, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('wrote', a.out)
