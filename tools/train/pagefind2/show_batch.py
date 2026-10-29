"""show_batch.py OUT.jpg [--synth S.npz --real R.npz --neg N.npz] [--zoom P --quad_band F]: one training batch as the
trainer builds it (zoom, augmentation, targets), drawn: the photo, the paper target outlined green, the band where the
4-corner labels' outline does not count shaded red, the edge target yellow. For checking augmentation changes."""
import argparse
import sys

import cv2
import numpy as np
import torch

sys.argv, argv = sys.argv[:1], sys.argv[1:]
import train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--synth', default='C:/fs_train/pf2/synth_c.npz')
ap.add_argument('--real', default='C:/fs_train/pf2/train_real.npz'); ap.add_argument('--neg', default='')
ap.add_argument('--zoom', type=float, default=0.5); ap.add_argument('--quad_band', type=float, default=0.035)
ap.add_argument('--n', type=int, default=6); ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args(argv)
dev = 'cuda'
gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
packs = [(T.GpuPack(a.synth, dev), a.n, 0), (T.GpuPack(a.real, dev), a.n, 1)] + ([(T.GpuPack(a.neg, dev), 2, 0)] if a.neg else [])
xs, ms, vs, bs, qs = [], [], [], [], []
for pk, n, q in packs:
    idx = rng.integers(0, pk.n, n).tolist()
    x, m, v, b = pk.get(idx)
    xs.append(x); ms.append(m); vs.append(v); bs.append(b); qs += [q] * n
x, m, v, b = torch.cat(xs), torch.cat(ms), torch.cat(vs), torch.cat(bs)
gt = torch.zeros_like(m)
B = x.shape[0]
x, m, gt = T.zoom(x, m, v, gt, b, np.nonzero(rng.random(B) < a.zoom)[0].tolist(), 0.55, gen)
x, m, v, gt = T.augment(x, m, v, gt, gen)
e = T.edge_target(m, v)
q = torch.tensor(qs, device=dev, dtype=torch.bool)
band = torch.zeros_like(m)
band[q] = T.quad_band(m[q], v[q], a.quad_band)
tiles = []
for k in range(B):
    im = (x[k].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
    bd = band[k, 0].cpu().numpy() > 0
    im[bd] = (0.5 * im[bd] + 0.5 * np.array([40, 40, 230])).astype(np.uint8)
    im[e[k, 0].cpu().numpy() > 0] = (0, 230, 230)
    cs, _ = cv2.findContours((m[k, 0].cpu().numpy() > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cs, -1, (40, 200, 40), 1)
    cv2.putText(im, 'real' if qs[k] else 'synth/neg', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(im)
while len(tiles) % 4:
    tiles.append(np.full_like(tiles[0], 255))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print('wrote', a.out, B, 'samples')
