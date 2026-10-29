"""eval_real_ckpts.py: SD-HR checkpoints on real photos with flat scans (dir300, DocReal; evaluation only, never
trained on), net only: UVDoc pass 1 (the shipped ONNX) -> the app's crop -> SD-HR -> unwarp -> MS-SSIM / LD
(train_sdhr.RealVal, as train_run2.py's --real_sets check).

    python eval_real_ckpts.py CKPT [CKPT ...] [--sets dir300,docreal] [--every 1] [--out results.json]

A CKPT is a trainer checkpoint (its EMA 'state_dict') or a plain state_dict file."""
import argparse
import json
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, '..'), os.path.join(HERE, '..', '..', 'flatnet'), os.path.join(HERE, '..', 'flat')):
    sys.path.insert(0, p)
import common as C  # noqa: E402
from train_sdhr import RealVal  # noqa: E402


def load(path, dev):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, 'sd.pt')
        torch.save({'state_dict': sd}, tmp)
        return C.load_sdhr(tmp, dev), ck.get('step')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--sets', default='dir300,docreal')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--root', default='E:/fs_datasets')
    ap.add_argument('--uvdoc_onnx', default=os.path.join(HERE, '..', '..', '..', 'app', 'src', 'main', 'assets', 'models', 'uvdoc.onnx'))
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vals = {s: RealVal([s], a.every, a.uvdoc_onnx, a.root) for s in a.sets.split(',')}
    res = {}
    for path in a.ckpts:
        net, step = load(path, dev)
        r = {}
        for s, rv in vals.items():
            t = time.time()
            ms, ld = rv(net, dev)
            r[s] = dict(ms_ssim=round(ms, 4), ld=round(ld, 3), n=len(rv.items))
            print(f'{os.path.basename(path)} (step {step}) {s}: MS-SSIM {ms:.4f}  LD {ld:.3f}  ({len(rv.items)} photos, {time.time() - t:.0f} s)', flush=True)
        res[path] = r
        del net
        torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
