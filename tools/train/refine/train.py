"""train.py ARM OUT [STEPS]: the refinement pilot (round 20). ARM 'local' or 'context'.

Input: the stage-1 page (make_data.py in/), full resolution. Output: a correction d (frame px) per 32-px cell, the
refined map being B(x + d(x)) (targets.py). Trained on 768 x 768 crops (24 x 24 cells). The loss compares d's
differences between cells 1, 2, 4, 8 and 16 apart, not d itself: a constant shift of the page is not the
refiner's job, and a crop cannot know it. 'context' also sees the whole page at 384 x 384 (squashed) and where the
crop sits in it.

Data: E:/fs_train/refine/train/w*/ (index.jsonl, in/, t/); held-out test: E:/fs_train/refine/test.
"""
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL = 32
CROP = 24            # cells
CTX = 384
SCALE = 16.0         # the net outputs d / SCALE
STEPS_K = (1, 2, 4, 8, 16)
MAX_STRETCH = 0.3    # stage 1 lost the page's shape (targets.py stats): left out
MIN_VALID = 0.3


def rows_of(dirs):
    out = []
    for d in dirs:
        ip = os.path.join(d, 'index.jsonl')
        if not os.path.exists(ip):
            continue
        for line in open(ip):
            r = json.loads(line)
            s = r.get('stats')
            if s and s['stretch_p90'] <= MAX_STRETCH and s['valid_share'] >= MIN_VALID \
                    and os.path.exists(os.path.join(d, 't', r['id'] + '.npz')):
                r['dir'] = d
                out.append(r)
    return out


def load(r):
    img = cv2.imread(os.path.join(r['dir'], 'in', r['id'] + '.jpg'))
    with np.load(os.path.join(r['dir'], 't', r['id'] + '.npz')) as t:
        d, valid = t['d'], t['valid']
    ny, nx = valid.shape
    pad = np.zeros((ny * CELL, nx * CELL, 3), np.uint8)
    h, w = min(img.shape[0], ny * CELL), min(img.shape[1], nx * CELL)
    pad[:h, :w] = img[:h, :w]
    return pad, d, valid


def to_tensor(img):
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float() / 255.0


class Crops(torch.utils.data.Dataset):
    def __init__(self, rows, n, seed=0):
        self.rows, self.n, self.seed = rows, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k))
        img, d, valid = load(self.rows[rng.integers(len(self.rows))])
        ny, nx = valid.shape
        ctx = cv2.resize(img, (CTX, CTX), interpolation=cv2.INTER_AREA)
        # a crop whose cells are mostly on the page
        for _ in range(8):
            cy, cx = rng.integers(0, ny - CROP + 1), rng.integers(0, nx - CROP + 1)
            if valid[cy:cy + CROP, cx:cx + CROP].mean() > 0.5:
                break
        crop = img[cy * CELL:(cy + CROP) * CELL, cx * CELL:(cx + CROP) * CELL]
        dd, vv = d[cy:cy + CROP, cx:cx + CROP].copy(), valid[cy:cy + CROP, cx:cx + CROP].copy()
        # the crop's cell centres in the page, [-1, 1] (grid_sample order x, y)
        gx = (cx + np.arange(CROP) + 0.5) / nx * 2 - 1
        gy = (cy + np.arange(CROP) + 0.5) / ny * 2 - 1
        pos = np.stack(np.meshgrid(gx, gy), -1).astype(np.float32)
        # flips: the field flips with the page
        if rng.random() < 0.5:
            crop, ctx, dd, vv, pos = crop[:, ::-1], ctx[:, ::-1], dd[:, ::-1] * [-1, 1], vv[:, ::-1], pos[:, ::-1] * [-1, 1]
        if rng.random() < 0.5:
            crop, ctx, dd, vv, pos = crop[::-1], ctx[::-1], dd[::-1] * [1, -1], vv[::-1], pos[::-1] * [1, -1]
        x, c = to_tensor(np.ascontiguousarray(crop)), to_tensor(np.ascontiguousarray(ctx))
        # brightness / contrast / colour cast
        g = torch.tensor(rng.uniform(0.8, 1.2, 3), dtype=torch.float32)[:, None, None]
        b = float(rng.uniform(-0.08, 0.08))
        x, c = (x * g + b).clamp(0, 1), (c * g + b).clamp(0, 1)
        return (x, c, torch.from_numpy(np.ascontiguousarray(pos, np.float32)),
                torch.from_numpy(np.ascontiguousarray(dd, np.float32)).permute(2, 0, 1),
                torch.from_numpy(np.ascontiguousarray(vv)))


def block(ci, co, stride=1, dil=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, dil, dil, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


def encoder():
    """3 x H x W -> 96 x H/32 x W/32"""
    return nn.Sequential(block(3, 16, 2), block(16, 32, 2), block(32, 32), block(32, 48, 2), block(48, 48),
                         block(48, 64, 2), block(64, 64), block(64, 96, 2), block(96, 96), block(96, 96, dil=2))


class Refiner(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.enc = encoder()
        cin = 96
        if context:
            self.cenc = nn.Sequential(encoder(), block(96, 64))          # 384 -> 12 x 12
            self.cglob = nn.Sequential(block(64, 64, 2), block(64, 64, 2), nn.AdaptiveAvgPool2d(1))
            cin += 128 + 2
        self.head = nn.Sequential(block(cin, 96), block(96, 96, dil=2), block(96, 64), nn.Conv2d(64, 2, 1))

    def forward(self, x, ctx=None, pos=None):
        f = self.enc(x)
        if self.context:
            cf = self.cenc(ctx)
            local = F.grid_sample(cf, pos, mode='bilinear', align_corners=False)
            glob = self.cglob(cf).expand(-1, -1, f.shape[2], f.shape[3])
            f = torch.cat([f, local, glob, pos.permute(0, 3, 1, 2)], 1)
        return self.head(f) * SCALE


def diff_loss(pred, tgt, valid):
    """L1 of the difference of (pred - tgt) between cells k apart, along x and y, where both cells are valid"""
    e = pred - tgt
    v = valid.float()[:, None]
    tot, n = 0.0, 0.0
    for k in STEPS_K:
        if k >= e.shape[3]:
            break
        for dim in (2, 3):
            a = e.narrow(dim, k, e.shape[dim] - k) - e.narrow(dim, 0, e.shape[dim] - k)
            m = v.narrow(dim, k, v.shape[dim] - k) * v.narrow(dim, 0, v.shape[dim] - k)
            tot = tot + ((a.abs().sum(1, keepdim=True) / math.sqrt(k)) * m).sum()
            n = n + m.sum()
    return tot / n.clamp(min=1)


def page_inputs(img, valid):
    ny, nx = valid.shape
    x = to_tensor(img)[None]
    c = to_tensor(cv2.resize(img, (CTX, CTX), interpolation=cv2.INTER_AREA))[None]
    gx = (np.arange(nx) + 0.5) / nx * 2 - 1
    gy = (np.arange(ny) + 0.5) / ny * 2 - 1
    pos = torch.from_numpy(np.stack(np.meshgrid(gx, gy), -1).astype(np.float32))[None]
    return x, c, pos


def affine_free(e, valid):
    """e (ny x nx x 2) minus its best affine over the valid cells"""
    ny, nx = valid.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    U = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)], 1)
    m = valid.ravel()
    E = e.reshape(-1, 2)
    M, *_ = np.linalg.lstsq(U[m], E[m], rcond=None)
    return (E - U @ M).reshape(ny, nx, 2)


def page_metrics(e, valid):
    """RMS px after the best affine, and median / p90 stretch (|difference between neighbour cells| / CELL)"""
    r = affine_free(e, valid)
    n = np.linalg.norm(r[valid], axis=-1)
    sx = np.linalg.norm(np.diff(e, axis=1), axis=-1)[valid[:, 1:] & valid[:, :-1]]
    sy = np.linalg.norm(np.diff(e, axis=0), axis=-1)[valid[1:] & valid[:-1]]
    st = np.concatenate([sx, sy]) / CELL
    return float(np.sqrt(np.mean(n ** 2))), float(np.median(st)), float(np.percentile(st, 90))


@torch.no_grad()
def evaluate(net, rows, dev):
    """per page: (kind, before rms/stretch, after rms/stretch); 'before' is no correction (pred 0)"""
    net.eval()
    res = []
    for r in rows:
        img, d, valid = load(r)
        x, c, pos = page_inputs(img, valid)
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            p = net(x.to(dev), c.to(dev), pos.to(dev)).float()[0].permute(1, 2, 0).cpu().numpy()
        p = p[:valid.shape[0], :valid.shape[1]]
        res.append((r['kind'], page_metrics(-d, valid), page_metrics(p - d, valid)))
    net.train()
    return res


def summary(res):
    kinds = sorted({k for k, *_ in res})
    lines = []
    for k in kinds + ['ALL']:
        s = [(b, a) for kk, b, a in res if k == 'ALL' or kk == k]
        b, a = np.array([x[0] for x in s]), np.array([x[1] for x in s])
        lines.append(f'{k:12s} n {len(s):3d} | rms {np.median(b[:, 0]):5.2f} -> {np.median(a[:, 0]):5.2f} px'
                     f' | stretch med {100 * np.median(b[:, 1]):5.2f} -> {100 * np.median(a[:, 1]):5.2f} %'
                     f' | p90 {100 * np.median(b[:, 2]):5.2f} -> {100 * np.median(a[:, 2]):5.2f} %'
                     f' | better {(a[:, 2] < b[:, 2]).sum()} / worse {(a[:, 2] > b[:, 2]).sum()}')
    return '\n'.join(lines)


def main():
    arm, out = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    bs = int(os.environ.get('RF_BS', 12))
    os.makedirs(out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_rows = rows_of(sorted(glob.glob('E:/fs_train/refine/train/w*')))
    test_rows = rows_of(['E:/fs_train/refine/test'])
    print(f'{arm}: {len(train_rows)} training pages, {len(test_rows)} test pages, batch {bs}, {steps} steps', flush=True)
    net = Refiner(arm == 'context').to(dev)
    print('params', sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')
    dl = torch.utils.data.DataLoader(Crops(train_rows, steps * bs, seed=int(time.time())), batch_size=bs,
                                     num_workers=int(os.environ.get('RF_WORKERS', 6)), persistent_workers=True,
                                     prefetch_factor=4)
    t0, run = time.time(), 0.0
    for step, (x, c, pos, d, v) in enumerate(dl, 1):
        x, c, pos, d, v = x.to(dev), c.to(dev), pos.to(dev), d.to(dev), v.to(dev)
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
            p = net(x, c, pos)
        loss = diff_loss(p.float(), d, v)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            print(f'step {step} loss {run:.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.3f} s/step', flush=True)
        if step % 2500 == 0 or step == steps:
            torch.save(net.state_dict(), os.path.join(out, 'last.pt'))
            print(summary(evaluate(net, test_rows, dev)), flush=True)
        if step >= steps:
            break


if __name__ == '__main__':
    main()
