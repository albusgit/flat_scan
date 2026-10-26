"""train_sdhr.py: fine-tune SyntheticDoc-HR (the app's first flattening net) on exact ground truth, made the way the
app feeds it, and keep what it already does well.

  python train_sdhr.py --synth DIR[,DIR] --out RUN [--uvdoc PREP] [--anchor DIR] [--steps 30000] [--bs 8]

Why (measured 2026-10-26, round 17):
  - SD-HR invents curl on flat paper (a dollar bill came out bowed: the app now needs a flat-paper rule), adds wavy
    text on nearly flat pages, cuts curled corners off (the coupon's "SAVE $1" read "$7"), and was trained on A4
    only; on dir300 the app's SD-HR-first choice scores MS-SSIM 0.499 where UVDoc alone scores 0.548 and the best of
    the existing methods per photo 0.582.
  - the Round 11 SD-HR fine-tune on pseudo ground truth got better on its own validation and worse in the app.
Data (each batch mixes, --mix):
  synth   make_data.py: exact ground truth for flat / nearly flat / curled / corner-lifted / folded paper in the
          papers people scan (A4, letter, receipts, bills, cards, coupons, books), as the app crops it
  uvdoc   UVDoc_final (MIT): 20k photos of real captured paper shapes with true grids (prep_uvdoc.py), cropped the
          app's way; low resolution (712 x 488), so a minority
  anchor  real photos without ground truth (licence-clean sets): the original net's own output is the target
          (learning without forgetting): what it does well on real content stays
Loss (grid in [-1, 1]): L1 + w_d x L1 of the first differences (page-scale derivatives: waviness, local stretch)
  + w_c x L1 of the second differences on flat / nearly flat samples (a flat sheet's map is nearly linear)
  + w_b x L1 on the border rows / columns (where the page ends: cut corners, background taken in)
  + w_f x fold-over (cells whose orientation flips) + w_a x L1 to the original net on anchor samples.
Selection: every --eval steps, (1) held-out synthetic per kind (grid error, derivative error) and (2) the app's two
passes on real photos with flat scans (dir300 / DocReal, realval): UVDoc pass 1 (the shipped ONNX) -> crop -> this
net -> unwarp -> MS-SSIM / LD. best.pt = best real MS-SSIM among checkpoints that are no worse than the start on
any synthetic kind (within --tol). The final word is the app pipeline on Windows (ft.sh + review.py, by eye).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'flatnet'))
sys.path.insert(0, os.path.join(HERE, '..', 'flat'))
import fast  # noqa: E402
import common as C  # noqa: E402  (tools/flatnet: SD-HR model, BM-only forward)

cv2.setNumThreads(1)
GH, GW = C.GH, C.GW
IN_H, IN_W = C.IN_H, C.IN_W
FLATLIKE = ('flat', 'nearflat')


# ---------------------------------------------------------------- data

def turned_grid(g):
    """a page grid G (2 x GH x GW, rows = page t, cols = page s) -> the grid as the net reads a page it saw turned
    90 deg cw: net row i = page s = i / (GH - 1), net col j = page t = 1 - j / (GW - 1) (uvdoc.cpp UvGrid::at)"""
    s = np.linspace(0, 1, GH)[:, None] * (GW - 1)          # column index in G
    t = (1 - np.linspace(0, 1, GW))[None, :] * (GH - 1)    # row index in G
    mx = np.broadcast_to(s, (GH, GW)).astype(np.float32)
    my = np.broadcast_to(t, (GH, GW)).astype(np.float32)
    return np.stack([cv2.remap(g[k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for k in range(2)])


def app_crop(img, grid, rng, margin=(0.02, 0.10), jitter=0.02):
    """img: the photo, grid: 2 x GH x GW photo px of the page (rows = t). -> (RGB IN_H x IN_W, target, turned)"""
    H, W = img.shape[:2]
    c = np.array([grid[:, 0, 0], grid[:, 0, -1], grid[:, -1, -1], grid[:, -1, 0]])
    ext = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
    c = c + rng.uniform(-jitter, jitter, c.shape) * ext
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    m = rng.uniform(*margin) * max(x1 - x0, y1 - y0)
    bx0, by0 = max(0, int(math.floor(x0 - m))), max(0, int(math.floor(y0 - m)))
    bx1, by1 = min(W, int(math.ceil(x1 + m)) + 1), min(H, int(math.ceil(y1 + m)) + 1)
    turned = np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[3] - c[0])
    crop = img[by0:by1, bx0:bx1]
    t = (turned_grid(grid) if turned else grid) - np.array([bx0, by0], np.float32)[:, None, None]
    cw, ch = bx1 - bx0, by1 - by0
    if turned:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        t = np.stack([(ch - 1) - t[1], t[0]])
        cw, ch = ch, cw
    x = cv2.resize(crop, (IN_W, IN_H), interpolation=cv2.INTER_AREA if cw > IN_W else cv2.INTER_LINEAR)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB), np.stack([t[0] / (cw - 1) * 2 - 1, t[1] / (ch - 1) * 2 - 1]).astype(np.float32), turned


class Mixed(Dataset):
    """one epoch = `length` samples drawn by --mix from the sources; returns (x uint8 3xHxW, target 2xGHxGW,
    kind id, is_anchor)"""
    KINDS = ['flat', 'nearflat', 'curl', 'corner', 'fold', 'fold2', 'curl_corner', 'wave_curl', 'uvdoc', 'anchor']

    def __init__(self, synth_dirs, uvdoc, anchor, mix, length, held=False, seed=0):
        self.synth = []
        for d in synth_dirs:
            for line in open(os.path.join(d, 'index.jsonl')):
                r = json.loads(line)
                r['root'] = d
                self.synth.append(r)
        # held-out: every 50th synthetic sample (by id) never trains
        self.synth = [r for r in self.synth if (int(r['id'].split('_')[0]) % 50 == 7) == held]
        self.uv_items, self.uv_grids = [], None
        if uvdoc:
            items = json.load(open(os.path.join(uvdoc, 'index.json')))
            self.uv_grids = np.load(os.path.join(uvdoc, 'grids.npy'), mmap_mode='r')
            self.uv_root = uvdoc
            self.uv_items = [(i, it) for i, it in enumerate(items) if it['val'] == held]
        self.anchor = []
        if anchor:
            for line in open(os.path.join(anchor, 'index.jsonl')):
                r = json.loads(line)
                if not os.path.isabs(r['photo']):
                    r['photo'] = os.path.join(anchor, r['photo'])
                self.anchor.append(r)
        w = {'synth': mix[0] if self.synth else 0, 'uvdoc': mix[1] if self.uv_items else 0, 'anchor': mix[2] if self.anchor else 0}
        tot = sum(w.values())
        self.p = {k: v / tot for k, v in w.items()}
        self.length, self.held, self.seed, self.epoch = length, held, seed, 0

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i, int(self.held)))
        src = rng.choice(list(self.p), p=list(self.p.values()))
        if self.held:   # deterministic: walk the held synthetic set
            src = 'synth'
        if src == 'synth':
            r = self.synth[i % len(self.synth)] if self.held else self.synth[rng.integers(len(self.synth))]
            p = os.path.join(r['root'], r['path'])
            x = cv2.cvtColor(cv2.imread(p + '.jpg'), cv2.COLOR_BGR2RGB)
            t = np.load(p + '.npy').astype(np.float32)
            kind = self.KINDS.index(r['kind'])
            anchor = 0
        elif src == 'uvdoc':
            k, it = self.uv_items[rng.integers(len(self.uv_items))]
            img = cv2.imread(os.path.join(self.uv_root, it['img']))
            g = np.asarray(self.uv_grids[k], np.float32)                     # 2 x 89 x 61 photo px (rows = t)
            g = np.stack([cv2.resize(g[c], (GW, GH), interpolation=cv2.INTER_LINEAR) for c in range(2)])
            x, t, _ = app_crop(img, g, rng)
            kind, anchor = self.KINDS.index('uvdoc'), 0
        else:
            a = self.anchor[rng.integers(len(self.anchor))]
            img = cv2.imread(a['photo'])
            # the page finder's corners stand in for pass 1: a coarse page grid from them (bilinear), then the app crop
            q = np.array(a['corners'], np.float32)   # TL TR BR BL
            s = np.linspace(0, 1, GW)[None, :, None]
            tt = np.linspace(0, 1, GH)[:, None, None]
            g = (q[0] * (1 - s) * (1 - tt) + q[1] * s * (1 - tt) + q[2] * s * tt + q[3] * (1 - s) * tt).transpose(2, 0, 1)
            x, t, _ = app_crop(img, g.astype(np.float32), rng)
            t = np.zeros_like(t)   # filled by the teacher on the GPU
            kind, anchor = self.KINDS.index('anchor'), 1
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1), torch.from_numpy(t), kind, anchor


def gpu_aug(x, gen):
    """x: B x 3 x H x W uint8 -> float in [-1, 1] with colour jitter, blur and noise (geometry stays exact)"""
    B = x.shape[0]
    dev = x.device
    x = x.float() / 255
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*(s or (B,)), device=dev, generator=gen)
    x = (x * u(0.75, 1.2).view(B, 1, 1, 1) + u(-0.06, 0.06).view(B, 1, 1, 1)).clamp(0, 1) ** u(0.8, 1.25).view(B, 1, 1, 1)
    g = x.mean(1, keepdim=True)
    x = g + (x - g) * u(0.6, 1.3).view(B, 1, 1, 1)
    x = x + torch.randn(x.shape, device=dev, generator=gen) * u(0, 0.02).view(B, 1, 1, 1)
    return x.clamp(0, 1) * 2 - 1


# ---------------------------------------------------------------- losses

def diffs(g):
    """first differences along page columns (u) and rows (v), in page units (the grid spans 2 per side)"""
    du = (g[..., :, 1:] - g[..., :, :-1]) * (GW - 1) / 2
    dv = (g[..., 1:, :] - g[..., :-1, :]) * (GH - 1) / 2
    return du, dv


def losses(p, t, kind, anchor):
    """per-sample terms; p, t: B x 2 x GH x GW"""
    l1 = (p - t).abs().mean(dim=(1, 2, 3))
    pu, pv = diffs(p)
    tu, tv = diffs(t)
    ld = (pu - tu).abs().mean(dim=(1, 2, 3)) + (pv - tv).abs().mean(dim=(1, 2, 3))
    puu, pvv = pu[..., :, 1:] - pu[..., :, :-1], pv[..., 1:, :] - pv[..., :-1, :]
    tuu, tvv = tu[..., :, 1:] - tu[..., :, :-1], tv[..., 1:, :] - tv[..., :-1, :]
    lc = (puu - tuu).abs().mean(dim=(1, 2, 3)) + (pvv - tvv).abs().mean(dim=(1, 2, 3))
    e = (p - t).abs()
    lb = torch.cat([e[:, :, 0, :], e[:, :, -1, :]], 2).mean(dim=(1, 2)) + torch.cat([e[:, :, :, 0], e[:, :, :, -1]], 2).mean(dim=(1, 2))
    # fold-over: the cell's orientation (cross product of its two edge vectors) against the target's
    pu_, pv_ = pu[:, :, :-1, :], pv[:, :, :, :-1]
    tu_, tv_ = tu[:, :, :-1, :], tv[:, :, :, :-1]
    cp = pu_[:, 0] * pv_[:, 1] - pu_[:, 1] * pv_[:, 0]
    ct = tu_[:, 0] * tv_[:, 1] - tu_[:, 1] * tv_[:, 0]
    lf = F.relu(-cp * torch.sign(ct)).mean(dim=(1, 2))
    return l1, ld, lc, lb, lf


# ---------------------------------------------------------------- real-photo check (the app's two passes)

class RealVal:
    """dir300 / DocReal photos with flat scans: UVDoc pass 1 (the shipped ONNX) -> the app's crop -> SD-HR (the model
    being trained) -> unwarp -> MS-SSIM / LD against the scan (tools/train/flat/realval.py metrics)"""

    def __init__(self, sets, every, uvdoc_onnx, root):
        import onnxruntime as ort
        import realval as rv
        self.rv = rv
        self.items = []
        for s in sets:
            self.items += rv.load(os.path.join(root, 'sets', s + '.json'), every, root)
        prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.uv = ort.InferenceSession(uvdoc_onnx, providers=prov)
        # pass 1 once: the crop boxes do not depend on the net being trained
        self.crops = []
        for _, img, ref in self.items:
            h, w = img.shape[:2]
            g = self.uv.run(None, {'image': rv.net_input(img)})[0][0]
            gx, gy = rv.to_px(g, w, h)
            c = np.array([[gx[0, 0], gy[0, 0]], [gx[0, -1], gy[0, -1]], [gx[-1, -1], gy[-1, -1]], [gx[-1, 0], gy[-1, 0]]])
            x0, y0 = c.min(0)
            x1, y1 = c.max(0)
            m = 0.06 * max(x1 - x0, y1 - y0)
            box = (max(0, int(np.floor(x0 - m))), max(0, int(np.floor(y0 - m))), min(w, int(np.ceil(x1 + m)) + 1), min(h, int(np.ceil(y1 + m)) + 1))
            turned = np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[3] - c[0])
            self.crops.append((box, turned))

    @torch.no_grad()
    def __call__(self, net, dev):
        from concurrent.futures import ThreadPoolExecutor
        outs = []
        for (_, img, ref), (box, turned) in zip(self.items, self.crops):
            bx0, by0, bx1, by1 = box
            crop = img[by0:by1, bx0:bx1]
            if turned:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            ch, cw = crop.shape[:2]
            x = cv2.cvtColor(cv2.resize(crop, (IN_W, IN_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            xt = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev).float() / 127.5 - 1
            with fast.bf16():
                g = C.bm_only_forward(net, xt)[0].float().cpu().numpy()
            # net grid -> crop px -> the page as the net saw it, turned back
            gx, gy = (g[0] + 1) / 2 * (cw - 1), (g[1] + 1) / 2 * (ch - 1)
            oh, ow = (ref.shape[1], ref.shape[0]) if turned else ref.shape[:2]
            mx = cv2.resize(gx.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
            my = cv2.resize(gy.astype(np.float32), (ow, oh), interpolation=cv2.INTER_LINEAR)
            page = cv2.remap(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), mx, my, cv2.INTER_LINEAR)
            if turned:
                page = cv2.rotate(page, cv2.ROTATE_90_COUNTERCLOCKWISE)
            outs.append((ref, page))
        torch.set_num_threads(1)
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda rp: (self.rv.ms_ssim(rp[0], rp[1]), self.rv.ld(rp[0], rp[1])), outs))
        return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))


@torch.no_grad()
def synth_eval(net, loader, dev):
    """per kind: grid L1 (x1000) and derivative error (x1000) on held-out synthetic samples"""
    acc = {}
    for x, t, kind, _ in loader:
        x = x.to(dev).float() / 127.5 - 1
        with fast.bf16():
            p = C.bm_only_forward(net, x).float()
        l1, ld, _, _, _ = losses(p, t.to(dev), None, None)
        for k, a, b in zip(kind.tolist(), l1.tolist(), ld.tolist()):
            n = Mixed.KINDS[k]
            s = acc.setdefault(n, [0, 0.0, 0.0])
            s[0] += 1; s[1] += a; s[2] += b
    return {k: (v[1] / v[0] * 1000, v[2] / v[0] * 1000) for k, v in acc.items()}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--uvdoc', default='')
    ap.add_argument('--anchor', default='')
    ap.add_argument('--mix', default='0.65,0.2,0.15', help='synth, uvdoc, anchor shares of a batch')
    ap.add_argument('--init', default=C.SD_CKPT)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--w', default='1,0.5,0.5,1,2,1', help='weights: l1, deriv, curvature (flat kinds), border, fold, anchor')
    ap.add_argument('--freeze_enc', type=int, default=0)
    ap.add_argument('--eval', type=int, default=2000)
    ap.add_argument('--real_sets', default='dir300,docreal')
    ap.add_argument('--real_every', type=int, default=3)
    ap.add_argument('--uvdoc_onnx', default=os.path.join(HERE, '..', '..', '..', 'app', 'src', 'main', 'assets', 'models', 'uvdoc.onnx'))
    ap.add_argument('--tol', type=float, default=0.03, help='a synthetic kind may get this much worse (relative)')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--no_compile', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = fast.setup(0)
    w = [float(v) for v in a.w.split(',')]
    mix = [float(v) for v in a.mix.split(',')]
    sdirs = a.synth.split(',')
    tr = Mixed(sdirs, a.uvdoc, a.anchor, mix, a.steps * a.bs // 20 + a.bs)
    va = Mixed(sdirs, '', '', [1, 0, 0], 0, held=True)
    va.length = min(len(va.synth), 800)
    print(f'train: synth {len(tr.synth)}, uvdoc {len(tr.uv_items)}, anchor {len(tr.anchor)}; mix {tr.p}; held synth {va.length}', flush=True)
    tl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=a.bs, shuffle=False, num_workers=min(4, a.workers))
    net = C.load_sdhr(a.init, dev)
    teacher = C.load_sdhr(a.init, dev).eval() if tr.anchor else None
    for name, p in net.named_parameters():
        p.requires_grad_(not (name.startswith('shd_') or (a.freeze_enc and name.startswith(('enc_s1', 'enc_s2', 'enc_s3')))))
    params = [p for p in net.parameters() if p.requires_grad]
    opt = fast.adamw(params, a.lr)
    sched = fast.onecycle(opt, a.lr, a.steps, min(500, a.steps // 20))
    ema = fast.Ema(net, 0.999)
    fwd = C.bm_only_forward   # (torch.compile of the module: see README; eager is fine on one GPU)
    log = fast.Log(os.path.join(a.out, 'log.jsonl'))
    real = RealVal(a.real_sets.split(','), a.real_every, a.uvdoc_onnx, fast.data_root()) if a.real_sets else None
    base_syn = synth_eval(ema.model, vl, dev)
    base_real = real(ema.model, dev) if real else (0, 0)
    log(step=0, real_ms_ssim=base_real[0], real_ld=base_real[1], **{f'syn_{k}': v[0] for k, v in base_syn.items()},
        **{f'synd_{k}': v[1] for k, v in base_syn.items()})
    best = -1
    gen = torch.Generator(device=dev)
    gen.manual_seed(1)
    step, ep = 0, 0
    while step < a.steps:
        tr.epoch = ep
        ep += 1
        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()   # BatchNorm keeps the 1M-sample statistics
        for x, t, kind, anc in tl:
            x = gpu_aug(x.to(dev, non_blocking=True), gen)
            t = t.to(dev, non_blocking=True)
            anc = anc.to(dev).bool()
            if teacher is not None and anc.any():
                with torch.no_grad(), fast.bf16():
                    t[anc] = C.bm_only_forward(teacher, x[anc]).float()
            with fast.bf16():
                p = fwd(net, x).float()
            l1, ld, lc, lb, lf = losses(p, t, kind, anc)
            flat = torch.tensor([Mixed.KINDS[k] in FLATLIKE for k in kind.tolist()], device=dev).float()
            wa = torch.where(anc, torch.full_like(l1, w[5]), torch.ones_like(l1))
            loss = (wa * (w[0] * l1 + w[1] * ld + w[2] * flat * lc + w[3] * lb) + w[4] * lf).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            ema.update(net)
            step += 1
            if step % 200 == 0:
                log(step=step, loss=loss.item(), l1=l1.mean().item(), ld=ld.mean().item(), lb=lb.mean().item(), lr=sched.get_last_lr()[0])
            if step % a.eval == 0 or step == a.steps:
                syn = synth_eval(ema.model, vl, dev)
                rl = real(ema.model, dev) if real else (0, 0)
                ok = all(syn[k][0] <= base_syn[k][0] * (1 + a.tol) for k in base_syn if k in syn)
                log(step=step, real_ms_ssim=rl[0], real_ld=rl[1], synth_ok=ok, **{f'syn_{k}': v[0] for k, v in syn.items()},
                    **{f'synd_{k}': v[1] for k, v in syn.items()})
                ck = {'state_dict': ema.model.state_dict(), 'step': step, 'real': rl, 'synth': syn, 'args': vars(a)}
                torch.save(ck, os.path.join(a.out, f'step{step:06d}.pt'))
                if ok and rl[0] > best:
                    best = rl[0]
                    torch.save(ck, os.path.join(a.out, 'best.pt'))
                    log(step=step, best=best)
            if step >= a.steps:
                break
    print('done; best real MS-SSIM', best, 'start', base_real, flush=True)


if __name__ == '__main__':
    main()
