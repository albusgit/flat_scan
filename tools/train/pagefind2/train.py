"""Page finder v2 trainer (round 19). Everything after the RAM packs runs on the GPU: JPEG decode (nvjpeg), mask
unpacking, augmentation, the gutter teacher, the edge target, so the CPU stays free for the synthetic generator.

Targets: paper = the pack's mask; edge = a ~2 px band on the mask's outline away from the photo's own border (a page
cut by the frame has no edge there); gutter = the round-4 page finder's gutter output on the same photo (its only
job here is to keep book-spread splitting working; flagged: that model's training data is not licence-clean).

  python train.py --synth S.npz --real R.npz --val V.npz [V2.npz ...] --out DIR [--steps N] [speed switches]
  python train.py ... --bench 200      time 200 steps after 50 warm-up, print steps/s and the mean loss, no saving
Speed switches: --amp none|bf16  --compile 0|1|max  --channels_last 0|1  --bs N  --fused 0|1  --tf32 0|1"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from net import MEAN, STD, SIZE, SegNet2  # noqa: E402
import data as D  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'gpugen'))
import jpeg  # noqa: E402  (GPU JPEG where nvjpeg works, else the CPU)

ROUND4 = os.environ.get('FS_ROUND4', 'E:/fs_datasets/round4')   # the rented box: its own copy (vast/)
IMAGENET = os.path.join(ROUND4, 'assets', 'weights', 'mobilenetv3_large_100_ra-f55367f5.pth')
TEACHER = os.path.join(ROUND4, 'work', 'runs', 'run1', 'best.pt')


class GpuPack:
    def __init__(self, path, dev):
        z = D.load(path)
        self.n = len(z['box'])
        self.jpeg = torch.from_numpy(z['jpeg'])
        self.off = z['off']
        self.mask = torch.from_numpy(z['mask']).to(dev)
        self.box = torch.from_numpy(z['box']).to(dev)
        self.src = z['src']
        self.gut = torch.from_numpy(z['gut']).to(dev) if 'gut' in z else None   # gutter.py: the teacher, precomputed
        self.sources = list(z['sources'])
        self.dev = dev
        self.shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)

    def get(self, idx):
        # nvjpeg decodes on its own stream into memory the caching allocator may still have in use by the queued
        # previous step: without this wait the weights went NaN within ~10 steps (round 19 bake-off; costs < 1 %)
        torch.cuda.synchronize()
        imgs = jpeg.decode([self.jpeg[self.off[i]:self.off[i + 1]] for i in idx], self.dev)
        x = torch.stack(imgs).float().div_(255)                                # B x 3 x S x S, RGB
        it = torch.as_tensor(idx, device=self.dev)
        bits = self.mask[it]
        m = ((bits[..., None] >> self.shifts) & 1).reshape(len(idx), 1, SIZE, SIZE).float()
        b = self.box[it]
        yy = torch.arange(SIZE, device=self.dev)[None, :, None]
        xx = torch.arange(SIZE, device=self.dev)[None, None, :]
        v = ((xx >= b[:, 0, None, None]) & (xx < b[:, 2, None, None]) & (yy >= b[:, 1, None, None]) & (yy < b[:, 3, None, None]))
        return x, m, v[:, None].float(), b

    def gutter(self, idx):
        g = self.gut[torch.as_tensor(idx, device=self.dev)][:, None].float() / 255
        return F.interpolate(g, size=(SIZE, SIZE), mode='bilinear', align_corners=False)


def augment(x, m, v, gt, gen):
    B = x.shape[0]
    dev = x.device
    # geometry: flips and quarter turns, applied to photo, mask and valid together (the box follows)
    k = int(torch.randint(0, 4, (1,), generator=gen, device=dev))
    if k:
        x, m, v, gt = (torch.rot90(t, k, (2, 3)) for t in (x, m, v, gt))
    if bool(torch.rand(1, generator=gen, device=dev) < 0.5):
        x, m, v, gt = (torch.flip(t, (3,)) for t in (x, m, v, gt))
    # colour: brightness, contrast, saturation, gamma, a colour cast; sometimes grey, blur, noise
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, 1, 1, generator=gen, device=dev)
    g = x.mean(1, keepdim=True)
    x = g + (x - g) * r(0.6, 1.4)
    x = (x - 0.5) * r(0.7, 1.3) + 0.5 + r(-0.12, 0.12)
    x = x * (1 + 0.08 * (torch.rand(B, 3, 1, 1, generator=gen, device=dev) * 2 - 1))
    x = x.clamp(0, 1) ** r(0.75, 1.35)
    grey = (torch.rand(B, 1, 1, 1, generator=gen, device=dev) < 0.1).float()
    x = grey * x.mean(1, keepdim=True).expand_as(x) + (1 - grey) * x
    blur = (torch.rand(B, 1, 1, 1, generator=gen, device=dev) < 0.25).float()
    x = blur * F.avg_pool2d(x, 3, 1, 1, count_include_pad=False) + (1 - blur) * x
    x = x + torch.randn(x.shape, generator=gen, device=dev) * r(0, 0.03)
    x = x.clamp(0, 1)
    pad = torch.tensor(MEAN, device=dev)[None, :, None, None]
    x = v * x + (1 - v) * pad                       # padding stays the mean colour, as the app pads
    return x, m, v, gt


def zoom(x, m, v, gt, b, sel, smin, gen):
    """zoom into the photo for the samples in `sel`: a window of the same shape as the photo (scale U(smin, 1) of it,
    anywhere inside it) stretched back over the photo's box, so pages fill, overfill or are cut by the frame. The box
    (v) stays; the mask is resampled and thresholded."""
    if not len(sel):
        return x, m, gt
    dev = x.device
    i = torch.as_tensor(sel, device=dev)
    n = len(sel)
    bb = b[i].float()
    s = smin + (1 - smin) * torch.rand(n, generator=gen, device=dev)
    wh = bb[:, 2:] - bb[:, :2]
    c0 = bb[:, :2] + torch.rand(n, 2, generator=gen, device=dev) * wh * (1 - s[:, None])
    t = c0 - s[:, None] * bb[:, :2]                     # letterbox px: q = s * p + t
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = s; theta[:, 1, 1] = s
    theta[:, :, 2] = s[:, None] - 1 + (1 - s[:, None] + 2 * t) / SIZE   # the same map in grid_sample's [-1, 1] units
    grid = F.affine_grid(theta, (n, 1, SIZE, SIZE), align_corners=False)
    x, m, gt = x.clone(), m.clone(), gt.clone()
    x[i] = F.grid_sample(x[i], grid, mode='bilinear', padding_mode='border', align_corners=False)
    m[i] = (F.grid_sample(m[i], grid, mode='bilinear', padding_mode='zeros', align_corners=False) > 0.5).float() * v[i]
    gt[i] = F.grid_sample(gt[i], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return x, m, gt


def quad_band(m, v, frac, widths=(2, 4, 6, 9, 12, 16)):
    """the band along a 4-corner label's outline where the true paper edge may lie (a curved side bows off the
    straight line between the corners): half-width frac x the page's diagonal (from its area), 2..16 px. Along the
    photo's own border the paper is not eroded (no outline there)."""
    def pool(t, w):   # square max filter as two 1-D passes
        t = F.max_pool2d(t, (1, 2 * w + 1), 1, (0, w))
        return F.max_pool2d(t, (2 * w + 1, 1), 1, (w, 0))
    want = (frac * torch.sqrt(2 * m.sum((1, 2, 3)))).clamp(2, widths[-1])
    out = torch.zeros_like(m)
    inner = torch.maximum(m, 1 - v)
    for k, w in enumerate(widths):
        lo = widths[k - 1] if k else 0
        pick = (want > lo) & (want <= w)
        if pick.any():
            band = pool(m[pick], w) - (-pool(-inner[pick], w))
            out[pick] = band
    return out


def edge_target(m, v):
    band = F.max_pool2d(m, 3, 1, 1) - (-F.max_pool2d(-m, 3, 1, 1))
    inner = -F.max_pool2d(-v, 7, 1, 3)              # 3 px inside the photo: its own border is no paper edge
    return band * inner


def normalise(x):
    mean = torch.tensor(MEAN, device=x.device)[None, :, None, None]
    std = torch.tensor(STD, device=x.device)[None, :, None, None]
    return (x - mean) / std


@torch.no_grad()
def teacher_gutter(teacher, x, b):
    """the round-4 net on each photo as it ran (the valid box squashed to 256): its gutter probability, back in the box"""
    crops = []
    for k in range(x.shape[0]):
        x0, y0, x1, y1 = (int(t) for t in b[k])
        crops.append(F.interpolate(x[k:k + 1, :, y0:y1, x0:x1], size=(256, 256), mode='bilinear', align_corners=False, antialias=True))
    lg = teacher(normalise(torch.cat(crops)))[:, 1:2].float()
    out = torch.zeros(x.shape[0], 1, SIZE, SIZE, device=x.device)
    for k in range(x.shape[0]):
        x0, y0, x1, y1 = (int(t) for t in b[k])
        out[k, :, y0:y1, x0:x1] = torch.sigmoid(F.interpolate(lg[k:k + 1], size=(y1 - y0, x1 - x0), mode='bilinear', align_corners=False))
    return out


def loss_fn(out, m, v, e, gt, keep=None, delta_px=None):
    """keep: where the paper / edge targets count (v minus the band along 4-corner labels' outlines); None = v.
    delta_px (B, px): channel 0 as a signed distance with the interval loss (sdf_loss) instead of BCE + Dice"""
    out = out.float()
    keep = v if keep is None else keep
    pm, pg, pe = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    if delta_px is not None:
        sl = sdf_loss(pm, past_frame(m, v), keep, delta_px)
        gl = (F.binary_cross_entropy_with_logits(pg, gt, pos_weight=torch.tensor(15.0, device=pg.device), reduction='none') * v).sum((1, 2, 3)) / v.sum((1, 2, 3))
        el = (F.binary_cross_entropy_with_logits(pe, e, pos_weight=torch.tensor(8.0, device=pe.device), reduction='none') * keep).sum((1, 2, 3)) / keep.sum((1, 2, 3)).clamp_min(1)
        loss = sl.mean() + 0.3 * gl.mean() + 0.5 * el.mean()
        return loss, dict(sdf=sl.mean().detach(), gut=gl.mean().detach(), edge=el.mean().detach())
    wide = F.max_pool2d(m, 7, 1, 3) - (-F.max_pool2d(-m, 7, 1, 3))
    wgt = (1 + 4 * wide) * keep
    bce = (F.binary_cross_entropy_with_logits(pm, m, reduction='none') * wgt).sum((1, 2, 3)) / wgt.sum((1, 2, 3)).clamp_min(1)
    sp = torch.sigmoid(pm) * keep
    mk = m * keep
    dice = 1 - (2 * (sp * mk).sum((1, 2, 3)) + 1) / (sp.sum((1, 2, 3)) + mk.sum((1, 2, 3)) + 1)
    gl = (F.binary_cross_entropy_with_logits(pg, gt, pos_weight=torch.tensor(15.0, device=pg.device), reduction='none') * v).sum((1, 2, 3)) / v.sum((1, 2, 3))
    el = (F.binary_cross_entropy_with_logits(pe, e, pos_weight=torch.tensor(8.0, device=pe.device), reduction='none') * keep).sum((1, 2, 3)) / keep.sum((1, 2, 3)).clamp_min(1)
    loss = bce.mean() + 0.5 * dice.mean() + 0.3 * gl.mean() + 0.5 * el.mean()
    return loss, dict(bce=bce.mean().detach(), dice=dice.mean().detach(), gut=gl.mean().detach(), edge=el.mean().detach())


# ---------------------------------------------------------------- signed distance and the interval loss (round 19)
# Channel 0 can be trained as the signed distance to the paper's outline (px at SIZE, + inside), which the app reads
# as logits (sign = paper; sigmoid = a 1 px soft edge; bicubic upsampling gives sub-pixel outlines). A 4-corner label
# says where the paper is up to delta of its straight sides (curved sides bow off them; the corners are exact): the
# loss only penalises predictions outside [d_quad - delta, d_quad + delta]. delta per source, as a share of the page's
# diagonal: exact labels (synthetic, SAM outlines kept by sam_label.py and sam_clean.py) 0; SAM outlines with some
# wobble (sam_clean.py: NAME_samr) ~5 px at 384; rigid things (cards, IDs, passports) and
# SmartDoc's flat sheets small; paper and receipts wide.
SDF_D = 16.0
DELTA = [('_samr', 0.01), ('_sam', 0.004), ('synth', 0.0), ('negcrop', 0.0), ('smartdoc', 0.006), ('card', 0.006), ('id', 0.006),
         ('ktp', 0.006), ('passport', 0.006), ('midv', 0.006), ('receipt', 0.04)]
DELTA_DEFAULT = 0.03


def source_delta(name):
    for key, d in DELTA:
        if key in name:
            return d
    return DELTA_DEFAULT


def _erode(m, cross):
    if cross:   # 4-neighbour erosion
        p = F.pad(m, (1, 1, 1, 1), value=1.0)
        return torch.minimum(torch.minimum(p[..., 1:-1, :-2], p[..., 1:-1, 2:]), torch.minimum(p[..., :-2, 1:-1], p[..., 2:, 1:-1])).minimum(m)
    return -F.max_pool2d(-F.pad(m, (1, 1, 1, 1), value=1.0), 3, 1, 0)


def sdf(m, D=SDF_D):
    """B x 1 x H x W 0/1 mask -> signed distance (px, + inside, the outline at 0), clipped to +-D. Octagonal distance
    (alternate 8- and 4-neighbour erosions: within ~8 % of Euclidean; the zero crossing is exact)."""
    def inside(a):
        d = torch.zeros_like(a)
        cur = a
        for k in range(int(D) + 1):   # a fixed count: no wait for the GPU
            d = d + cur
            cur = _erode(cur, cross=k % 2 == 1)
        return d
    # pixel centres: the first pixel inside lies 0.5 px from the outline
    return (inside(m) - 0.5 * m - inside(1 - m) + 0.5 * (1 - m)).clamp(-D, D)


def past_frame(m, v):
    """the mask carried on past the photo's border (each pixel outside the photo's box takes the nearest border
    pixel's value): a page cut by the frame then has no outline at the frame in its signed distance, it goes on
    straight, as the frame cuts it (the frame is no paper edge: edge_target leaves it out too)"""
    B, _, H, W = m.shape
    vx, vy = v.amax(2)[:, 0] > 0, v.amax(3)[:, 0] > 0                       # B x W, B x H
    aw, ah = torch.arange(W, device=m.device), torch.arange(H, device=m.device)
    x0, x1 = torch.where(vx, aw, W - 1).min(1).values, torch.where(vx, aw, 0).max(1).values
    y0, y1 = torch.where(vy, ah, H - 1).min(1).values, torch.where(vy, ah, 0).max(1).values
    xi = torch.minimum(torch.maximum(aw[None], x0[:, None]), x1[:, None].clamp_min(x0[:, None]))
    yi = torch.minimum(torch.maximum(ah[None], y0[:, None]), y1[:, None].clamp_min(y0[:, None]))
    idx = (yi[:, :, None] * W + xi[:, None, :]).reshape(B, 1, -1)
    return m.reshape(B, 1, -1).gather(2, idx).reshape(B, 1, H, W)


def sdf_loss(pm, m, keep, delta_px):
    """pm: predicted channel 0 (px); m: target mask (the quad's fill for 4-corner labels); keep: pixels that count;
    delta_px: B (px) -> per-sample mean of the interval L1 + the sign's BCE where the sign is certain"""
    d = sdf(m)
    dp = delta_px[:, None, None, None]
    lo, hi = (d - dp).clamp(-SDF_D, SDF_D), (d + dp).clamp(-SDF_D, SDF_D)
    p = pm.clamp(-SDF_D, SDF_D)
    err = F.relu(p - hi) + F.relu(lo - p)
    l1 = (err * keep).sum((1, 2, 3)) / keep.sum((1, 2, 3)).clamp_min(1)
    sure = keep * (d.abs() > dp + 1).float()
    bce = (F.binary_cross_entropy_with_logits(pm, (d > 0).float(), reduction='none') * sure).sum((1, 2, 3)) / sure.sum((1, 2, 3)).clamp_min(1)
    return l1 / 4 + bce


class GenRing:
    """synthetic samples from tools/train/gpugen on the GPU, kept in a ring of `size` (uint8 photo, packed in memory as
    the packs are): each step adds `new` fresh samples and draws from the whole ring (each sample used about
    size / new times over its life, with fresh augmentation each time), so a slow generator never stalls training"""

    def __init__(self, size, new, seed, dev, gen_dev='', **kw):
        sys.path.insert(0, os.path.join(HERE, '..', 'gpugen'))
        self.size, self.new, self.dev = size, new, dev
        if gen_dev:     # the generator in its own process (gpugen/feeder.py): fresh samples as fast as it makes them
            from feeder import Feeder
            self.f, self.g = Feeder('pf', max(8, new), gen_dev, seed, **kw), None
        else:
            from gen import GpuGen
            self.f, self.g = None, GpuGen(seed=seed, dev=dev, **kw)
        self.x = torch.zeros(size, 3, SIZE, SIZE, dtype=torch.uint8, device=dev)
        self.m = torch.zeros(size, 1, SIZE, SIZE, dtype=torch.bool, device=dev)
        self.gut = torch.zeros(size, 1, SIZE, SIZE, dtype=torch.bool, device=dev)   # exact gutter lines (book spreads)
        self.box = torch.zeros(size, 4, dtype=torch.long, device=dev)
        self.n, self.head, self.made = 0, 0, 0
        while self.n < size:
            if self.f:
                self.put(self.f.take_one(dev))
            else:
                self.add(min(32, size - self.n))

    def add(self, k):
        b = self.g.batch(k)
        self.put(dict(x=(b['x'] * 255).round().to(torch.uint8), m=b['m'] > 0.5, gut=b['gut'] > 0.5, box=b['box']))

    def put(self, b):
        k = len(b['x'])
        idx = (self.head + torch.arange(k, device=self.dev)) % self.size
        self.x[idx], self.m[idx], self.gut[idx], self.box[idx] = b['x'], b['m'], b['gut'], b['box']
        self.made += k
        self.head = (self.head + k) % self.size
        self.n = min(self.size, self.n + k)

    def gutter(self, idx):
        """the gutter target of samples idx (call after get(idx): the same ring state)"""
        g = self.gut[torch.as_tensor(idx, device=self.dev)].float()
        return F.max_pool2d(g, 3, 1, 1)   # ~3 px wide, as the teacher's soft line

    def get(self, idx):
        if self.f:
            for b in self.f.take(self.dev):
                self.put(b)
        elif self.new:
            self.add(self.new)
        it = torch.as_tensor(idx, device=self.dev)
        x = self.x[it].float() / 255
        b = self.box[it]
        yy = torch.arange(SIZE, device=self.dev)[None, :, None]
        xx = torch.arange(SIZE, device=self.dev)[None, None, :]
        v = ((xx >= b[:, 0, None, None]) & (xx < b[:, 2, None, None]) & (yy >= b[:, 1, None, None]) & (yy < b[:, 3, None, None]))
        return x, self.m[it].float(), v[:, None].float(), b


def start_hint(a):
    """the step a --resume continues from (a resumed run's generator gets a fresh seed)"""
    if not a.resume or not os.path.exists(a.resume):
        return 0
    try:
        return json.load(open(os.path.join(os.path.dirname(a.resume), 'last.json')))['step']
    except (OSError, ValueError, KeyError):
        return 1


@torch.no_grad()
def evaluate(net, pack, amp, bs=16):
    """per sample IoU of the predicted paper (sigmoid > 0.5, inside the photo) with the target. Batches of the
    training's size, the last one padded (cudnn.benchmark tunes every new shape: odd batches cost ~75 s an evaluation)"""
    net.eval()
    ious = []
    for lo in range(0, pack.n, bs):
        idx = list(range(lo, min(pack.n, lo + bs)))
        n = len(idx)
        idx += [idx[-1]] * (bs - n)
        x, m, v, b = pack.get(idx)
        x = v * x + (1 - v) * torch.tensor(MEAN, device=x.device)[None, :, None, None]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            out = net(normalise(x))
        p = (out[:, 0:1].float() > 0) & (v > 0)
        t = (m > 0.5) & (v > 0)
        inter = (p & t).sum((1, 2, 3)).float()
        union = (p | t).sum((1, 2, 3)).float()
        empty = union == 0
        ious += torch.where(empty, torch.ones_like(inter), inter / union.clamp_min(1))[:n].tolist()
    net.train()
    I = np.array(ious)
    per = {}
    for i, s in enumerate(pack.sources):
        sel = pack.src == i
        if sel.any():
            per[s] = (round(float(I[sel].mean()), 4), int((I[sel] < 0.9).sum()), int(sel.sum()))
    return float(I.mean()), int((I < 0.9).sum()), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', default=''); ap.add_argument('--real', default=''); ap.add_argument('--val', nargs='*', default=[])
    ap.add_argument('--out', default=''); ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--real_share', type=float, default=0.4); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--eval', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--amp', default='bf16'); ap.add_argument('--compile', default='0'); ap.add_argument('--channels_last', type=int, default=1)
    ap.add_argument('--fused', type=int, default=1); ap.add_argument('--tf32', type=int, default=1); ap.add_argument('--bench', type=int, default=0)
    ap.add_argument('--resume', default='')
    ap.add_argument('--stop_at', type=int, default=0, help='save and stop at this step (the schedule stays that of --steps)')
    ap.add_argument('--init', default='', help='start from these weights (a state_dict: step*.pt / best.pt)')
    ap.add_argument('--neg', default='', help='a pack of photos with no page (data.py negcrops)')
    ap.add_argument('--neg_share', type=float, default=0.0, help='share of the batch drawn from --neg')
    ap.add_argument('--quad_band', type=float, default=0.0,
                    help='real (4-corner) labels: no paper / edge loss within this share of the page diagonal of the outline')
    ap.add_argument('--zoom', type=float, default=0.0, help='share of samples zoomed in (zoom())')
    ap.add_argument('--zoom_min', type=float, default=0.55)
    ap.add_argument('--sdf', type=int, default=0, help='channel 0 as a signed distance, the interval loss per source (sdf_loss)')
    ap.add_argument('--gpugen_ring', type=int, default=0, help='synthetic samples from tools/train/gpugen in a ring of this size (replaces --synth)')
    ap.add_argument('--gpugen_new', type=int, default=4, help='fresh generated samples per step')
    ap.add_argument('--gen_dev', default='', help='the generator in its own process on this device (cuda:1: a GPU of its own): fresh samples as fast as it makes them, the step never waits for them')
    ap.add_argument('--no_teacher', type=int, default=0, help='no round-4 gutter teacher (its data is not licence-clean): '
                    'gutter targets from the generator book spreads only, none on real photos')
    ap.add_argument('--pages', default='C:/fs_train/pages2'); ap.add_argument('--bg_pack', default='C:/fs_train/pf2/train_neg2.npz')
    a = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32); torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.92)   # fail fast instead of spilling into system RAM (Windows WDDM)
    syn = GenRing(a.gpugen_ring, a.gpugen_new, a.seed + 7919 * (1 + start_hint(a)), dev, a.gen_dev, pages_dir=a.pages, bg_pack=a.bg_pack)         if a.gpugen_ring else GpuPack(a.synth, dev)
    real = GpuPack(a.real, dev) if a.real else None
    neg = GpuPack(a.neg, dev) if a.neg else None
    real_delta = [source_delta(n) for n in real.sources] if real else []
    vals = [GpuPack(v, dev) for v in a.val] if not a.bench else []
    net = SegNet2(IMAGENET).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev))
    mf = torch.channels_last if a.channels_last else torch.contiguous_format
    teacher = None
    if not a.no_teacher:   # (with --no_teacher the round-4 net is never loaded: the box needs no copy of it)
        sys.path.insert(0, os.path.join(ROUND4, 'seg'))
        import segnet as old
        teacher = old.SegNet().to(dev).eval()
        teacher.load_state_dict(torch.load(TEACHER, map_location=dev))
        teacher.requires_grad_(False)
        teacher = teacher.to(memory_format=mf)
    net = net.to(memory_format=mf)
    enc = list(net.enc.parameters()); encid = {id(p) for p in enc}
    dec = [p for p in net.parameters() if id(p) not in encid]
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.lr / 3}, {'params': dec, 'lr': a.lr}], weight_decay=1e-4, fused=bool(a.fused))
    base = [a.lr / 3, a.lr]
    fwd = net
    if a.compile != '0':
        fwd = torch.compile(net, mode='max-autotune' if a.compile == 'max' else None)
    amp = a.amp == 'bf16'
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    start, best = 0, -1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); start, best = ck['step'], ck['best']
        rng = np.random.default_rng(a.seed + start)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    steps = a.bench + 50 if a.bench else a.steps
    nneg = int(round(a.bs * a.neg_share)) if neg else 0
    nreal = int(round(a.bs * a.real_share)) if real else 0
    nsyn = a.bs - nreal - nneg
    t0, losses, skipped = time.time(), [], 0
    # BatchNorm statistics as of the last good step: a non-finite forward has already folded into them when its
    # step is skipped
    bn_bufs = [b for n, b in net.named_buffers() if b.is_floating_point()]
    bn_good = [b.clone() for b in bn_bufs]

    def save_last(step):
        ck = dict(net=net.state_dict(), opt=opt.state_dict(), step=step, best=best, args=vars(a), skipped=skipped)
        torch.save(ck, os.path.join(a.out, 'last.pt.tmp')); os.replace(os.path.join(a.out, 'last.pt.tmp'), os.path.join(a.out, 'last.pt'))
        with open(os.path.join(a.out, 'last.json'), 'w') as f:   # for overnight_r19.py: where a restart resumes
            json.dump(dict(step=step, best=best), f)
    for step in range(start + 1, steps + 1):
        if a.bench and step == 51:
            torch.cuda.synchronize(); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
            made0 = getattr(syn, 'made', 0)
        if not a.bench:
            warm = min(1.0, step / 500)
            for gi, gr in enumerate(opt.param_groups):
                gr['lr'] = base[gi] * warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        xs, ms, vs, bs, gs = [], [], [], [], []
        fr = []   # per sample: delta (share of the page diagonal) for --sdf
        for pk, n in ((syn, nsyn), (real, nreal), (neg, nneg)):
            if n:
                idx = rng.integers(0, pk.n, n).tolist()
                x, m, v, b = pk.get(idx)
                xs.append(x); ms.append(m); vs.append(v); bs.append(b)
                if getattr(pk, 'gut', None) is not None:
                    gs.append(pk.gutter(idx))
                elif a.no_teacher:   # real photos are single pages: no gutter (spreads come from the generator)
                    gs.append(torch.zeros_like(m))
                else:
                    gs.append(teacher_gutter(teacher, x, b))
                fr += [0.0] * n if pk is not real else [real_delta[pk.src[i]] for i in idx]
        x, m, v, b, gt = torch.cat(xs), torch.cat(ms), torch.cat(vs), torch.cat(bs), torch.cat(gs)
        if a.zoom:
            x, m, gt = zoom(x, m, v, gt, b, np.nonzero(rng.random(a.bs) < a.zoom)[0].tolist(), a.zoom_min, gen)
        x, m, v, gt = augment(x, m, v, gt, gen)
        e = edge_target(m, v)
        keep = v
        if a.quad_band and nreal:
            keep = v.clone()
            r = slice(nsyn, nsyn + nreal)
            keep[r] = v[r] * (1 - quad_band(m[r], v[r], a.quad_band))
        x = normalise(x).contiguous(memory_format=mf)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            out = fwd(x)
        dpx = None
        if a.sdf:
            dpx = torch.tensor(fr, device=dev) * torch.sqrt(2 * m.sum((1, 2, 3)))
        loss, parts = loss_fn(out, m, v, e, gt, keep, dpx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        if not torch.isfinite(gn):   # skip the step, put the statistics back (the step waits for the GPU anyway)
            skipped += 1
            opt.zero_grad(set_to_none=True)
            torch._foreach_copy_(bn_bufs, bn_good)
            print(json.dumps(dict(step=step, skipped=skipped, note='non-finite gradient: step skipped')), flush=True)
            if skipped > 50:
                raise SystemExit('more than 50 non-finite steps: stopping')
            continue
        opt.step()
        if step % 50 == 0:
            torch._foreach_copy_(bn_good, bn_bufs)
        if a.bench:
            if step > 50:
                losses.append(loss.detach())
            continue
        if step % 100 == 0:
            rec = dict(step=step, loss=round(loss.item(), 4), **{k: round(float(v_), 4) for k, v_ in parts.items()},
                       sec=round(time.time() - t0, 1), lr=opt.param_groups[1]['lr'], skipped=skipped, made=getattr(syn, 'made', 0))
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if a.out and (step % a.eval == 0 or step == steps):
            res = {}
            for vp, name in zip(vals, a.val):
                mi, nb, per = evaluate(net, vp, amp, a.bs)
                res[os.path.basename(name)] = dict(miou=round(mi, 4), fail=nb, n=vp.n, per=per)
            score = float(np.mean([r['miou'] for r in res.values()])) if res else -step
            rec = dict(step=step, eval=res, score=round(score, 4))
            print(json.dumps(rec)[:2000], flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
            torch.save(net.state_dict(), os.path.join(a.out, f'step{step:06d}.pt'))
            if score > best:
                best = score
                torch.save(net.state_dict(), os.path.join(a.out, 'best.pt'))
            save_last(step)
        elif a.out and step % 1000 == 0:   # a resume point between evaluations (a restart loses at most 1000 steps)
            save_last(step)
        if a.stop_at and step >= a.stop_at:
            save_last(step)
            print('stopped at', step, flush=True)
            break
    if a.bench:
        torch.cuda.synchronize()
        dt = time.time() - t0
        L = torch.stack(losses).float().cpu().numpy()
        print(json.dumps(dict(variant=dict(amp=a.amp, compile=a.compile, channels_last=a.channels_last, bs=a.bs, fused=a.fused, tf32=a.tf32),
                              steps_per_s=round(a.bench / dt, 3), samples_per_s=round(a.bench * a.bs / dt, 1),
                              peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), loss_mean=round(float(L.mean()), 4),
                              loss_last50=round(float(L[-50:].mean()), 4),
                              fresh_per_step=round((getattr(syn, 'made', 0) - made0) / a.bench, 2))), flush=True)


if __name__ == '__main__':
    main()
