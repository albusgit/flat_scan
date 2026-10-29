"""gutter.py PACK.npz [...]: the round-4 net's gutter output, computed once per photo and stored in the pack as
`gut` (N x 96 x 96 uint8, the whole letterbox; train.py upsamples it). Replaces the teacher run in every step."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
import train as T
import data as D

dev = 'cuda'
sys.path.insert(0, T.ROUND4 + '/seg')
import segnet as old  # noqa: E402
teacher = old.SegNet().to(dev).eval()
teacher.load_state_dict(torch.load(T.TEACHER, map_location=dev))
for path in sys.argv[1:]:
    pk = T.GpuPack(path, dev)
    out = np.zeros((pk.n, 96, 96), np.uint8)
    for lo in range(0, pk.n, 32):
        idx = list(range(lo, min(pk.n, lo + 32)))
        x, m, v, b = pk.get(idx)
        g = T.teacher_gutter(teacher, x, b)
        out[lo:lo + len(idx)] = (F.interpolate(g, size=(96, 96), mode='area')[:, 0] * 255).round().byte().cpu().numpy()
    z = D.load(path)
    z['gut'] = out
    np.savez(path + '.tmp.npz', **z)
    import os
    os.replace(path + '.tmp.npz', path)
    print(path, pk.n, 'gutter mean', out.mean(), flush=True)
