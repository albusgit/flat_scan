"""check_sdhr.py OUT.jpg [--n 8] [--dev cuda] [--bench N]: GpuGen SD-HR samples drawn for checking: the crop with every
8th target grid point (red), and the crop flattened by its own target (grid_sample): a right target gives the page
upright and straight. --bench: samples/s."""
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from gen import GpuGen

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--n', type=int, default=8); ap.add_argument('--dev', default='cuda')
ap.add_argument('--bench', type=int, default=0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--pool', type=int, default=64)
a = ap.parse_args()
g = GpuGen(mode='sdhr', dev=a.dev, pool=a.pool, seed=a.seed)
b = g.sdhr_batch(a.n)
if a.bench:
    sync = torch.cuda.synchronize if a.dev == 'cuda' else (lambda: None)
    sync(); t0 = time.time()
    for _ in range(a.bench):
        b = g.sdhr_batch(a.n)
    sync()
    print(f'SD-HR: {a.bench * a.n / (time.time() - t0):.1f} samples/s, peak GPU '
          f'{torch.cuda.max_memory_allocated() / 1e9 if a.dev == "cuda" else 0:.2f} GB', flush=True)
x = b['x'].float() / 255
flat = F.grid_sample(x, b['t'].permute(0, 2, 3, 1), mode='bilinear', align_corners=True)     # n x 3 x GH x GW
flat = F.interpolate(flat, size=(4 * 121, int(4 * 121 * 1.0 / 1.0 * 177 / 121)), mode='bilinear', align_corners=False)
tiles = []
for i in range(a.n):
    im = (x[i].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
    im = cv2.resize(im, (512, 720), interpolation=cv2.INTER_AREA).copy()
    t = b['t'][i].cpu().numpy()
    for r in range(0, t.shape[1], 8):
        for c in range(0, t.shape[2], 8):
            px = (t[0, r, c] + 1) / 2 * 511
            py = (t[1, r, c] + 1) / 2 * 719
            cv2.circle(im, (int(px), int(py)), 2, (40, 40, 230), -1)
    cv2.putText(im, f"{g.KINDS[int(b['kind'][i])]}{' turned' if b['turned'][i] else ''}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    f = (flat[i].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
    f = cv2.resize(f, (512, int(512 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
    f = np.vstack([f, np.full((720 - f.shape[0], 512, 3), 255, np.uint8)]) if f.shape[0] < 720 else f[:720]
    tiles.append(np.hstack([im, np.full((720, 6, 3), 0, np.uint8), f, np.full((720, 16, 3), 255, np.uint8)]))
rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles) - 1, 2)]
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print('wrote', a.out)
