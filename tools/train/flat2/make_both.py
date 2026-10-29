"""make_both.py PAGES OUT N [--workers W] [--seed S] [--pass1 uvdoc.onnx] [--cut P]: one synthetic photo -> training samples
for both nets (round 19), so the CPU renders each photo once.

  OUT/sdhr/      as make_data.py: s<seed>/<id>.jpg + .npy, manifests/, index.jsonl   (SD-HR: the app crop + its grid)
  OUT/pf/<seed>.npz   page finder, per job: img n x S x S x 3 uint8 (the whole photo letterboxed to S, BGR),
                      mask n x S x S uint8 (paper 0/255), valid n x S x S uint8 (255 inside the photo, 0 in the padding),
                      meta (json: kind, aspect, cut)
The page finder's input: the photo scaled so its long side is S, centred, padded with the ImageNet mean colour
(pagefind2/net.py letterbox, the same as the app will do). A share --cut of the photos frames the page cut by the
photo's edge (the finder must learn that such a side has no edge)."""
import argparse
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
S = 384
MEAN_BGR = (104, 116, 124)


def letterbox(img, size=S, interp=cv2.INTER_AREA, pad=MEAN_BGR):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.empty((size, size) + img.shape[2:], img.dtype)
    out[...] = pad if img.ndim == 3 else 0
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    valid = np.zeros((size, size), np.uint8)
    valid[y0:y0 + nh, x0:x0 + nw] = 255
    return out, valid


def init(pass1, cut):
    os.environ['FS_CUT_P'] = str(cut)
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(1)
    import make_data
    make_data.init(pass1)


def work(job):
    import make_data
    import synth
    synth.DEVICE = 'cpu'
    pages, out, seed, n = job
    rng = np.random.default_rng(seed)
    shard = os.path.join(out, 'sdhr', f's{seed:06d}')
    os.makedirs(shard, exist_ok=True)
    rows, pf_img, pf_mask, pf_valid, pf_meta = [], [], [], [], []
    i = 0
    while len(pf_img) < n and i < n * 6:
        i += 1
        try:
            sm = synth.sample(rng, pages)
        except Exception:
            continue
        if sm is None:
            continue
        im, valid = letterbox(sm['photo'])
        m, _ = letterbox((sm['mask'] * 255).astype(np.uint8), interp=cv2.INTER_AREA)
        pf_img.append(im); pf_mask.append(m); pf_valid.append(valid)
        pf_meta.append(dict(kind=sm['kind'], aspect=sm['aspect'], cut=sm['cut']))
        if sm['cut']:
            continue   # its page is not wholly in the photo: no SD-HR target
        u = rng.random()
        if make_data.PASS1 is not None and u < 0.3:
            crop, ai = 'pass1', synth.app_input(sm, rng, margin=(0.06, 0.06), jitter=0, corners=make_data.pass1_corners(sm['photo']))
        elif u < 0.5:
            crop, ai = 'wide', synth.app_input(sm, rng, margin=(0.06, 0.16), jitter=0.05)
        else:
            crop, ai = 'gt', synth.app_input(sm, rng)
        if ai is None:
            continue
        x, t, turned = ai
        if not np.isfinite(t).all() or np.abs(t).max() > 1:
            continue
        sid = f'{seed:06d}_{len(rows):04d}'
        cv2.imwrite(os.path.join(shard, sid + '.jpg'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        np.save(os.path.join(shard, sid + '.npy'), t.astype(np.float32))
        rows.append({'id': sid, 'path': f's{seed:06d}/{sid}', 'kind': sm['kind'], 'aspect': sm['aspect'],
                     'hw': round(float(sm['hw']), 4), 'turned': bool(turned), 'crop': crop})
    # compact: the photo as JPEG bytes, the mask bit-packed, the photo's box in the letterbox (pagefind2/data.py's pack)
    jp = [cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].ravel() for im in pf_img]
    boxes = []
    for v in pf_valid:
        ys, xs = np.nonzero(v)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    np.savez(os.path.join(out, 'pf', f'{seed:06d}.npz'), jpeg=np.concatenate(jp), off=np.concatenate([[0], np.cumsum([len(b) for b in jp])]),
             mask=np.stack([np.packbits((m > 127).ravel()) for m in pf_mask]), box=np.array(boxes, np.int32), meta=json.dumps(pf_meta))
    man = os.path.join(out, 'sdhr', 'manifests', f'{seed:06d}.json')
    with open(man + '.tmp', 'w') as f:
        json.dump({'seed': seed, 'rows': rows, 'attempts': i}, f)
    os.replace(man + '.tmp', man)
    return seed, len(rows), len(pf_img)


if __name__ == '__main__':
    import glob
    import time
    import make_data
    ap = argparse.ArgumentParser()
    ap.add_argument('pages'); ap.add_argument('out'); ap.add_argument('n', type=int)
    ap.add_argument('--workers', type=int, default=10); ap.add_argument('--seed', type=int, default=300000)
    ap.add_argument('--per_job', type=int, default=40); ap.add_argument('--pass1', default='')
    ap.add_argument('--cut', type=float, default=0.25)
    a = ap.parse_args()
    for d in ('sdhr/manifests', 'pf'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    pages = sorted(glob.glob(os.path.join(a.pages, '**', '*.png'), recursive=True) + glob.glob(os.path.join(a.pages, '**', '*.jpg'), recursive=True))
    njobs = (a.n + a.per_job - 1) // a.per_job
    jobs = [(pages, a.out, a.seed + k, a.per_job) for k in range(njobs)
            if not os.path.exists(os.path.join(a.out, 'pf', f'{a.seed + k:06d}.npz'))]
    t0, done_pf, done_sd = time.time(), 0, 0
    with Pool(a.workers, initializer=init, initargs=(a.pass1, a.cut)) as pool:
        for k, (seed, nsd, npf) in enumerate(pool.imap_unordered(work, jobs)):
            done_pf += npf; done_sd += nsd
            make_data.write_index(os.path.join(a.out, 'sdhr'), make_data.read_manifests(os.path.join(a.out, 'sdhr')))
            el = time.time() - t0
            print(f'job {k + 1}/{len(jobs)}: page finder {done_pf}, SD-HR {done_sd}, {done_pf / el:.2f} photos/s', flush=True)
