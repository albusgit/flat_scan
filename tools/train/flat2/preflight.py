"""preflight.py HELD [--ckpt C ...] [--n N]: before a run, how precisely the forward used in training gives SD-HR's grid.
Per checkpoint, on N held-out synthetic samples: the grid error (L1, derivative, curvature x 1000 against the target)
and the largest difference from the float32 reference (px of the 1024 x 1440 input) for
  bf16      the whole forward under bf16 autocast (run 1)
  fp32head  run2_core.bm_forward: bf16 encoder, float32 head (TF32 convolutions allowed, as fast.setup sets)
  fp32      everything float32, TF32 off (the reference: as the app runs the ONNX model)
Prints one line per (checkpoint, mode) and the seconds per batch of 8 (forward + backward) for bf16 / fp32head."""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'flatnet'), os.path.join(HERE, '..', 'flat')]
import common as C  # noqa: E402
from run2_core import bm_forward, map_terms  # noqa: E402


def run(net, x, mode):
    if mode == 'fp32':
        tf = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return C.bm_only_forward(net, x).float()
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf
    return bm_forward(net, x, fp32_head=mode == 'fp32head')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('held')
    ap.add_argument('--ckpt', nargs='*', default=[C.SD_CKPT])
    ap.add_argument('--n', type=int, default=200)
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = True
    rows = [json.loads(line) for line in open(os.path.join(a.held, 'index.jsonl'))][:a.n]
    for ck in a.ckpt:
        net = C.load_sdhr(ck, 'cuda')
        acc = {m: {'l1': [], 'deriv': [], 'curve': [], 'dpx': []} for m in ('bf16', 'fp32head', 'fp32')}
        for r in rows:
            p = os.path.join(a.held, r['path'])
            x = cv2.cvtColor(cv2.imread(p + '.jpg'), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.load(p + '.npy').astype(np.float32))[None].cuda()
            xt = (torch.from_numpy(x).permute(2, 0, 1).float().div(255) * 2 - 1)[None].cuda()
            with torch.no_grad():
                out = {m: run(net, xt, m) for m in acc}
            for m, g in out.items():
                q = map_terms(g, t)
                for k in ('l1', 'deriv', 'curve'):
                    acc[m][k].append(q[k].item() * 1000)
                d = (g - out['fp32']).abs() * 0.5 * torch.tensor([1023., 1439.], device='cuda')[None, :, None, None]
                acc[m]['dpx'].append(d.max().item())
        for m, v in acc.items():
            print(f'{os.path.basename(ck)} {m:9s} L1 {np.mean(v["l1"]):.2f}  deriv {np.mean(v["deriv"]):.2f}  curve {np.mean(v["curve"]):.2f}  '
                  f'max diff from fp32 {np.mean(v["dpx"]):.2f} px (mean over samples), {np.max(v["dpx"]):.2f} px (worst)', flush=True)
    # speed: forward + backward, batch 8
    net.train()
    x = torch.randn(8, 3, C.IN_H, C.IN_W, device='cuda')
    for m in ('bf16', 'fp32head'):
        for i in range(8):
            if i == 3:
                torch.cuda.synchronize(); t0 = time.time()
            bm_forward(net, x, fp32_head=m == 'fp32head').square().mean().backward()
        torch.cuda.synchronize()
        print(f'{m}: {(time.time() - t0) / 5:.3f} s per batch of 8 (forward + backward)', flush=True)


if __name__ == '__main__':
    main()
