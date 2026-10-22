"""Fine-tune SyntheticDoc-HR's backward-map head + encoder on real photos with pseudo ground truth.

  python train.py --data DIR[,DIR2] --out RUN_DIR [--epochs 12 --lr 2e-5 --al 0.0 --bs 4 ...]

Data: DIR/pairs.json + DIR/pgt/<id>.npz from pseudo_gt.py (grid in photo pixels). Each sample is cropped as the
app crops (core/src/uvdoc.cpp uv_crop/uv_input: the page's box + a margin of max(w, h) * U(m0, m1), clamped to
the photo, turned 90 deg clockwise when the page is wider than tall, INTER_AREA to 1024x1440).
BatchNorm stays frozen (running statistics of the 1M-sample training); the shadow decoder is not trained and not
run (the app does not use it; its weights are exported unchanged).
Loss: L1 on the grid (as the original training) + --al x axis-alignment loss (AADD, Wang et al. AAAI 2026:
variance along each grid row of the predicted points' v coordinate, along each column of u, in the target's
page coordinates; first-order inverse of the target grid)."""
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
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

cv2.setNumThreads(1)


def app_crop(bm_px, w, h, margin):
    """bm_px 2xGHxGW photo px. Returns box (x0, y0, x1, y1) and turned flag, as uv_crop (corners = grid corners)."""
    c = np.array([bm_px[:, 0, 0], bm_px[:, 0, -1], bm_px[:, -1, -1], bm_px[:, -1, 0]])  # TL TR BR BL of the page
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    m = margin * max(x1 - x0, y1 - y0)
    bx0, by0 = max(0, int(math.floor(x0 - m))), max(0, int(math.floor(y0 - m)))
    bx1, by1 = min(w, int(math.ceil(x1 + m)) + 1), min(h, int(math.ceil(y1 + m)) + 1)
    turned = np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[3] - c[0])
    return (bx0, by0, max(bx0 + 1, bx1), max(by0 + 1, by1)), bool(turned)


def make_input(img, bm_px, box, turned):
    """-> uint8 RGB 1440x1024 net input and the target grid in its [-1, 1] coords."""
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    t = bm_px.copy()
    t[0] -= x0
    t[1] -= y0
    cw, ch = x1 - x0, y1 - y0
    if turned:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        t = np.stack([(ch - 1) - t[1], t[0]])
        cw, ch = ch, cw
    small = cv2.resize(crop, (C.IN_W, C.IN_H), interpolation=cv2.INTER_AREA)
    return small, C.px_to_bm(t, cw, ch).astype(np.float32)


class PairSet(Dataset):
    def __init__(self, items, train, margins=(0.02, 0.10), seed=0):
        self.items, self.train, self.margins, self.seed = items, train, margins, seed
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rng = np.random.default_rng((self.seed, self.epoch, i) if self.train else (12345, i))
        img = np.load(it['cache'], mmap_mode='r')
        bm = np.load(it['cbm'])
        h, w = img.shape[:2]
        m = rng.uniform(*self.margins) if self.train else 0.06
        box, turned = app_crop(bm, w, h, m)
        if self.train:   # the page finder's corners are not exact: jitter the box by up to 2% of its size
            x0, y0, x1, y1 = box
            j = 0.02 * max(x1 - x0, y1 - y0)
            box = (int(np.clip(x0 + rng.uniform(-j, j), 0, w - 2)), int(np.clip(y0 + rng.uniform(-j, j), 0, h - 2)),
                   int(np.clip(x1 + rng.uniform(-j, j), 0, w)), int(np.clip(y1 + rng.uniform(-j, j), 0, h)))
            box = (box[0], box[1], max(box[0] + 16, box[2]), max(box[1] + 16, box[3]))
        x, t = make_input(np.asarray(img), bm, box, turned)
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        return x, torch.from_numpy(t)


def gpu_aug(x, t, gen, rot=6.0):
    """x: Bx3xHxW uint8 on the GPU, t: Bx2xGHxGW targets. Photometric jitter, blur, noise and a small in-plane
    rotation of the net input (targets moved with it). Returns x in [-1, 1]."""
    B, _, H, W = x.shape
    dev = x.device
    x = x.float() / 255.0
    u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*(s or (B,)), device=dev, generator=gen)
    on = (torch.rand(B, device=dev, generator=gen) < 0.8).float().view(B, 1, 1, 1)
    y = x * u(0.7, 1.2).view(B, 1, 1, 1) + u(-0.08, 0.08).view(B, 1, 1, 1)
    y = y.clamp(0, 1) ** u(0.75, 1.35).view(B, 1, 1, 1)
    g = y.mean(1, keepdim=True)
    y = g + (y - g) * u(0.6, 1.3).view(B, 1, 1, 1)
    y = y * u(0.9, 1.1, B, 3).view(B, 3, 1, 1)
    x = on * y + (1 - on) * x
    for b in range(B):
        if torch.rand(1, device=dev, generator=gen).item() < 0.25:
            sg = float(u(0.5, 1.6, 1).item())
            k = int(2 * round(3 * sg) + 1)
            x[b:b + 1] = TF.gaussian_blur(x[b:b + 1], [k, k], [sg, sg])
        if torch.rand(1, device=dev, generator=gen).item() < 0.3:
            x[b] = x[b] + torch.randn(x[b].shape, device=dev, generator=gen) * float(u(0.005, 0.03, 1).item())
    x = x.clamp(0, 1)
    if rot > 0:
        ang = u(-rot, rot) * math.pi / 180 * (torch.rand(B, device=dev, generator=gen) < 0.5).float()
        ca, sa = torch.cos(ang), torch.sin(ang)
        cx, cy = (W - 1) / 2, (H - 1) / 2
        # output pixel q -> input pixel p = R^-1 (q - c) + c
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        qx, qy = xx - cx, yy - cy
        px = ca.view(B, 1, 1) * qx + sa.view(B, 1, 1) * qy + cx
        py = -sa.view(B, 1, 1) * qx + ca.view(B, 1, 1) * qy + cy
        grid = torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        # targets: input px p -> output px q = R (p - c) + c
        tx = (t[:, 0] + 1) / 2 * (W - 1) - cx
        ty = (t[:, 1] + 1) / 2 * (H - 1) - cy
        nx = ca.view(B, 1, 1) * tx - sa.view(B, 1, 1) * ty + cx
        ny = sa.view(B, 1, 1) * tx + ca.view(B, 1, 1) * ty + cy
        t = torch.stack([nx / (W - 1) * 2 - 1, ny / (H - 1) * 2 - 1], 1)
    return x * 2 - 1, t


def al_loss(pred, gt):
    """AADD axis-alignment loss. pred, gt: Bx2xGHxGW grids (image coords). The predicted point at grid (i, j) is
    expressed in the target's page coordinates (u along columns j, v along rows i, in cells) with the target
    grid's local Jacobian; a straight row keeps v constant along j, a straight column u along i."""
    # Jacobian of gt w.r.t. (u=j, v=i) by central differences (one-sided at the border)
    du = torch.gradient(gt, dim=3)[0]          # d(x,y)/du
    dv = torch.gradient(gt, dim=2)[0]          # d(x,y)/dv
    a, b, c, d = du[:, 0], dv[:, 0], du[:, 1], dv[:, 1]   # J = [[a b], [c d]]
    det = a * d - b * c
    det = torch.where(det.abs() < 1e-8, torch.full_like(det, 1e-8), det)
    e = pred - gt
    ex, ey = e[:, 0], e[:, 1]
    eu = (d * ex - b * ey) / det                # J^-1 e
    ev = (-c * ex + a * ey) / det
    eu = eu.clamp(-5, 5)
    ev = ev.clamp(-5, 5)
    # in cells; normalise to page fractions so the scale does not depend on the grid size
    lh = ev.var(dim=2).mean() / (C.GH - 1) ** 2 * 4     # v variance along each row (dim 2 = columns j)
    lv = eu.var(dim=1).mean() / (C.GW - 1) ** 2 * 4     # u variance along each column
    return lh + lv


def load_items(dirs, stats_min_ncc, holdout, split):
    items = []
    for d in dirs:
        pairs = json.load(open(os.path.join(d, 'pairs.json')))
        st = json.load(open(os.path.join(d, 'pgt_stats.json')))
        for p in pairs:
            s = st.get(p['id'])
            if s is None or s['ncc2'] < stats_min_ncc or s['ncc2'] < s['ncc0']:
                continue
            items.append({'id': p['id'], 'cache': os.path.join(d, 'cache', p['id'] + '.npy'),
                          'cbm': os.path.join(d, 'cache', p['id'] + '_bm.npy')})
    rnd = random.Random(7)
    ids = sorted({it['id'] for it in items})
    hold = set(rnd.sample(ids, int(len(ids) * holdout)))
    if split == 'train':
        return [it for it in items if it['id'] not in hold]
    return [it for it in items if it['id'] in hold]


def evaluate(net, loader, dev):
    net.eval()
    tot, n, tot_al = 0.0, 0, 0.0
    with torch.no_grad():
        for x, t in loader:
            x = x.to(dev).float().div(127.5).sub(1)
            t = t.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = C.bm_only_forward(net, x)
            p = p.float()
            tot += F.l1_loss(p, t, reduction='sum').item() / (2 * C.GH * C.GW)
            tot_al += al_loss(p, t).item() * x.shape[0]
            n += x.shape[0]
    return tot / n, tot_al / n


def set_train_mode(net):
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--al', type=float, default=0.0)
    ap.add_argument('--rot', type=float, default=6.0)
    ap.add_argument('--min_ncc', type=float, default=0.5)
    ap.add_argument('--holdout', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--freeze_enc', type=int, default=0, help='freeze enc_s1..enc_s3')
    ap.add_argument('--init', default=C.SD_CKPT)
    ap.add_argument('--vram_gb', type=float, default=5.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30)))
    torch.manual_seed(0)
    dev = 'cuda'
    dirs = a.data.split(',')
    tr_items = load_items(dirs, a.min_ncc, a.holdout, 'train')
    va_items = load_items(dirs, a.min_ncc, a.holdout, 'val')
    print('train', len(tr_items), 'val', len(va_items), flush=True)
    tr = PairSet(tr_items, True)
    va = PairSet(va_items, False)
    tl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=False)
    vl = DataLoader(va, batch_size=a.bs, shuffle=False, num_workers=a.workers)
    net = C.load_sdhr(a.init, dev)
    train_params = []
    for name, p in net.named_parameters():
        frozen = name.startswith(('shd_',)) or (a.freeze_enc and name.startswith(('enc_s1', 'enc_s2', 'enc_s3')))
        p.requires_grad_(not frozen)
        if not frozen:
            train_params.append(p)
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(tl)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    v0 = evaluate(net, vl, dev)
    print('epoch 0 val l1 %.5f al %.6f' % v0, flush=True)
    log.write(json.dumps({'epoch': 0, 'val_l1': v0[0], 'val_al': v0[1]}) + '\n')
    log.flush()
    gen = torch.Generator(device=dev)
    gen.manual_seed(1)
    step = 0
    for ep in range(1, a.epochs + 1):
        tr.epoch = ep
        set_train_mode(net)
        t0, sl, sa, nb = time.time(), 0.0, 0.0, 0
        for x, t in tl:
            x, t = gpu_aug(x.to(dev, non_blocking=True), t.to(dev, non_blocking=True), gen, a.rot)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = C.bm_only_forward(net, x)
            p = p.float()
            l1 = F.l1_loss(p, t)
            loss = l1
            la = torch.zeros((), device=dev)
            if a.al > 0:
                la = al_loss(p, t)
                loss = loss + a.al * la
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            sched.step()
            step += 1
            sl += l1.item()
            sa += la.item()
            nb += 1
        v = evaluate(net, vl, dev)
        rec = {'epoch': ep, 'train_l1': sl / nb, 'train_al': sa / nb, 'val_l1': v[0], 'val_al': v[1],
               'sec': time.time() - t0, 'lr': sched.get_last_lr()[0]}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + '\n')
        log.flush()
        torch.save({'state_dict': net.state_dict(), 'epoch': ep, 'args': vars(a)}, os.path.join(a.out, f'ep{ep:02d}.pt'))


if __name__ == '__main__':
    main()
