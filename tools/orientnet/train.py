"""train.py PAGES OUT [--epochs E]: train orientnet on rendered upright pages (render_pages.py).

1. Patches: from every page, crops at random scales (text 6-40 px tall in the patch) where the ink share is text-like
   (orientnet.INK_MIN..INK_MAX), cached in PAGES/patches_{train,held}.npy.
2. Training: each patch turned by a random k x 90 degrees (the label: the turn that brings it back), with photo-like
   degradation (blur, noise, JPEG, uneven light, low contrast, small skew, rarely inverted).
3. Checkpoint by held-font accuracy; ONNX export OUT/orientnet.onnx (input N x 1 x 128 x 128, output N x 4 logits).
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import orientnet as on

P = on.PATCH


def cut_patches(files, per_page, seed):
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        got, tries = 0, 0
        while got < per_page and tries < per_page * 12:
            tries += 1
            # the patch covers P / s page px: s in [0.3, 1.3] (the pages' body text is 14-52 px tall)
            s = float(np.exp(rng.uniform(np.log(0.3), np.log(1.3))))
            side = int(P / s)
            if side >= min(g.shape):
                continue
            y = rng.integers(0, g.shape[0] - side)
            x = rng.integers(0, g.shape[1] - side)
            crop = cv2.resize(g[y:y + side, x:x + side], (P, P), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            fr = on.ink_mask(crop).mean()
            if on.INK_MIN <= fr <= on.INK_MAX:
                out.append(crop)
                got += 1
    return np.stack(out)


def degrade(p, rng):
    """one uint8 patch, made look photographed"""
    x = p.astype(np.float32)
    if rng.random() < 0.6:   # small residual skew / perspective
        a = rng.uniform(-4, 4)
        M = cv2.getRotationMatrix2D((P / 2, P / 2), a, rng.uniform(0.9, 1.1))
        x = cv2.warpAffine(x, M, (P, P), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.7:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.6))
    # contrast and uneven light
    lo, hi = rng.uniform(0, 90), rng.uniform(150, 255)
    x = lo + (x / 255.0) * (hi - lo)
    if rng.random() < 0.6:
        gx, gy = np.meshgrid(np.linspace(-1, 1, P), np.linspace(-1, 1, P))
        x = x * (1 + rng.uniform(-0.35, 0.35) * gx + rng.uniform(-0.35, 0.35) * gy)
    x = x + rng.normal(0, rng.uniform(0, 10), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, b = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 95))])
        x = cv2.imdecode(b, cv2.IMREAD_GRAYSCALE)
    if rng.random() < 0.04:
        x = 255 - x
    return x


class Data(torch.utils.data.Dataset):
    """patches from a .npy file, memory-mapped (each DataLoader worker maps the same file: no copies)"""

    def __init__(self, path, aug, seed=0):
        self.path, self.aug, self.seed = path, aug, seed
        self.n = len(np.load(path, mmap_mode='r'))
        self.p = None
        self.epoch = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.p is None:
            self.p = np.load(self.path, mmap_mode='r')
        rng = np.random.default_rng((self.seed, self.epoch, i))
        k = int(rng.integers(0, 4))
        x = np.array(self.p[i])
        if self.aug:
            x = degrade(x, rng)
        # the patch turned k x 90 ccw: turning it k x 90 cw brings it back (label k)
        x = np.ascontiguousarray(np.rot90(x, k))
        return torch.from_numpy(on.to_input(x[None])[0]), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pages')
    ap.add_argument('out')
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--per_page', type=int, default=48)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    for split, per in (('train', a.per_page), ('held', 16)):
        f = os.path.join(a.pages, f'patches_{split}.npy')
        if not os.path.exists(f):
            t = time.time()
            arr = cut_patches(sorted(glob.glob(os.path.join(a.pages, split, '*.png'))), per, 7 if split == 'train' else 8)
            np.save(f, arr)
            print(f'{split}: {len(arr)} patches in {time.time() - t:.0f} s', flush=True)
        cache[split] = f
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr = Data(cache['train'], True)
    va = Data(cache['held'], True, seed=1)
    dl = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, persistent_workers=True,
                                     drop_last=True)
    dv = torch.utils.data.DataLoader(va, batch_size=512, num_workers=a.workers)
    net = on.OrientNet().to(dev)
    print('params', sum(p.numel() for p in net.parameters()), 'device', dev, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    best = 0
    for ep in range(a.epochs):
        tr.epoch = ep
        net.train()
        t, tl, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = F.cross_entropy(net(x), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y)
            n += len(y)
        net.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, y in dv:
                p = net(x.to(dev)).argmax(1).cpu()
                ok += (p == y).sum().item()
                tot += len(y)
        acc = ok / tot
        print(f'epoch {ep}: loss {tl / n:.4f}  held-font patch accuracy {acc:.4f}  ({time.time() - t:.0f} s)', flush=True)
        if acc >= best:
            best = acc
            torch.save(net.state_dict(), os.path.join(a.out, 'orientnet.pt'))
    net.load_state_dict(torch.load(os.path.join(a.out, 'orientnet.pt')))
    net.cpu().eval()
    torch.onnx.export(net, torch.zeros(1, 1, P, P), os.path.join(a.out, 'orientnet.onnx'), input_names=['patches'],
                      output_names=['logits'], dynamic_axes={'patches': {0: 'n'}, 'logits': {0: 'n'}}, opset_version=17,
                      dynamo=False)
    print('best held-font patch accuracy', best)


if __name__ == '__main__':
    main()
