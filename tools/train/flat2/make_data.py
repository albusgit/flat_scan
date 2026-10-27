"""make_data.py PAGES OUT N [--workers W] [--seed S] [--kinds k1,k2] [--pass1 uvdoc.onnx]: pre-generate SD-HR
training samples.

Each sample (synth.sample + synth.app_input) is stored as
  OUT/s<seed>/<id>.jpg      the net's input as the app makes it: 1024 x 1440 RGB crop (JPEG q 93)
  OUT/s<seed>/<id>.npy      the target grid, float32 2 x 121 x 177 in [-1, 1] of that input (float16 rounding was
                            a floor of 0.013 under the derivative loss)
  OUT/manifests/<seed>.json one per finished job, written whole (rename): its samples ({"id", "path", "kind", "aspect",
                            "hw", "turned", "crop"}) and why attempts were left out
  OUT/index.jsonl           all finished jobs' samples (rewritten from the manifests as jobs finish)
N is the total wanted in OUT: a rerun skips the jobs whose manifest exists (an unfinished job's folder is redone).
The crop: 30 % from UVDoc pass 1 on the photo (--pass1; the app's crop with its real errors), 20 % with a wider,
uneven margin (pass 1 framing neighbouring things), the rest the page's corners jittered by 2 % + a 2-10 % margin.
A sample whose page does not lie wholly inside its crop is left out (its border would be unknowable).
Generation is CPU-bound (~1-3 s a sample a core). PAGES: a folder of flat page images (textures.py)."""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'flat'))
PASS1 = None


def init(pass1):
    global PASS1
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    if pass1:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        PASS1 = ort.InferenceSession(pass1, so, providers=['CPUExecutionProvider'])


def pass1_corners(photo):
    """UVDoc pass 1 on the whole photo (the shipped ONNX, realval's input and normalisation): its grid's corners"""
    import realval as rv
    h, w = photo.shape[:2]
    g = PASS1.run(None, {'image': rv.net_input(photo)})[0][0]
    gx, gy = rv.to_px(g, w, h)
    c = np.array([[gx[0, 0], gy[0, 0]], [gx[0, -1], gy[0, -1]], [gx[-1, -1], gy[-1, -1]], [gx[-1, 0], gy[-1, 0]]])
    return np.clip(c, [0, 0], [w - 1, h - 1])


def work(job):
    import synth
    synth.DEVICE = 'cpu'   # many processes: the CPU rasteriser each (the GPU is the trainer's)
    pages, out, seed, n, kinds = job
    rng = np.random.default_rng(seed)
    shard = os.path.join(out, f's{seed:06d}')
    os.makedirs(shard, exist_ok=True)
    rows, rej = [], {}
    synth.REJECT.clear()
    i = 0
    while len(rows) < n and i < n * 6:
        i += 1
        kind = kinds[rng.integers(len(kinds))] if kinds else None
        try:
            sm = synth.sample(rng, pages, kind)
        except Exception as e:   # a degenerate shape: skip it
            rej['error'] = rej.get('error', 0) + 1
            print('sample failed:', e, flush=True)
            continue
        if sm is None:
            continue
        u = rng.random()
        if PASS1 is not None and u < 0.3:
            crop = 'pass1'
            ai = synth.app_input(sm, rng, margin=(0.06, 0.06), jitter=0, corners=pass1_corners(sm['photo']))
        elif u < 0.5:
            crop = 'wide'
            ai = synth.app_input(sm, rng, margin=(0.06, 0.16), jitter=0.05)
        else:
            crop = 'gt'
            ai = synth.app_input(sm, rng)
        if ai is None:
            rej['empty_crop'] = rej.get('empty_crop', 0) + 1
            continue
        x, t, turned = ai
        if not np.isfinite(t).all() or np.abs(t).max() > 1:
            rej['outside_crop_' + crop] = rej.get('outside_crop_' + crop, 0) + 1
            continue
        sid = f'{seed:06d}_{len(rows):04d}'
        cv2.imwrite(os.path.join(shard, sid + '.jpg'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        np.save(os.path.join(shard, sid + '.npy'), t.astype(np.float32))
        rows.append({'id': sid, 'path': os.path.relpath(os.path.join(shard, sid), out).replace('\\', '/'), 'kind': sm['kind'],
                     'aspect': sm['aspect'], 'hw': round(float(sm['hw']), 4), 'turned': bool(turned), 'crop': crop})
    for k, v in synth.REJECT.items():
        rej[k] = rej.get(k, 0) + v
    man = os.path.join(out, 'manifests', f'{seed:06d}.json')
    with open(man + '.tmp', 'w') as f:
        json.dump({'seed': seed, 'rows': rows, 'attempts': i, 'rejected': rej}, f)
    os.replace(man + '.tmp', man)
    return seed, rows, rej


def read_manifests(out):
    rows = []
    for m in sorted(glob.glob(os.path.join(out, 'manifests', '*.json'))):
        rows += json.load(open(m))['rows']
    return rows


def write_index(out, rows):
    with open(os.path.join(out, 'index.jsonl.tmp'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(os.path.join(out, 'index.jsonl.tmp'), os.path.join(out, 'index.jsonl'))
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pages')
    ap.add_argument('out')
    ap.add_argument('n', type=int)
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per_job', type=int, default=50)
    ap.add_argument('--kinds', default='')
    ap.add_argument('--pass1', default='', help='UVDoc ONNX: 30 %% of the crops from its pass 1 on the photo')
    a = ap.parse_args()
    pages = sorted(glob.glob(os.path.join(a.pages, '**', '*.png'), recursive=True) +
                   glob.glob(os.path.join(a.pages, '**', '*.jpg'), recursive=True))
    os.makedirs(os.path.join(a.out, 'manifests'), exist_ok=True)
    kinds = [k for k in a.kinds.split(',') if k]
    done = {int(os.path.basename(m)[:6]) for m in glob.glob(os.path.join(a.out, 'manifests', '*.json'))}
    rows = read_manifests(a.out)
    have = len(rows)
    jobs, want, seed = [], a.n - have, a.seed
    while want > 0:   # jobs may come up short (rejections): N counts what is written; rerun to top up
        if seed not in done:
            jobs.append((pages, a.out, seed, min(a.per_job, want), kinds))
            want -= a.per_job
        seed += 1
    print(f'{have} samples there, {len(jobs)} jobs to run', flush=True)
    with Pool(a.workers, initializer=init, initargs=(a.pass1,)) as pool:
        tot = {}
        for seed, got, rej in pool.imap_unordered(work, jobs):
            rows += got
            for r, v in rej.items():
                tot[r] = tot.get(r, 0) + v
            print(f'{write_index(a.out, rows)} samples (job {seed}: {len(got)}; left out so far {tot})', flush=True)
    print(write_index(a.out, rows), 'samples')


if __name__ == '__main__':
    main()
