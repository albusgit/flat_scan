"""make_real.py INDEX OUT [--split train] [--max_per_source N] [--workers W] [--pass1 uvdoc.onnx]: real photos of flat
pages -> SD-HR training samples (round 19), in make_data.py's format (OUT/s<shard>/<id>.jpg + .npy, OUT/index.jsonl).

On a flat page the backward map is exactly the homography of its 4 corners, so a labelled quad gives an exact
target (normalised grid: no page size needed). Only pages whose paper edges are straight in the photo are kept:
along each side inside the photo, the strongest edge within +-2.5 % of the side's length must lie on the side's
line (median offset <= 0.3 %, 90th percentile <= 0.8 % of the page diagonal, and found on >= 70 % of the side).
Book pages and curled receipts fail this and stay out (their quad is not their shape). Licence: the index's
`licence == train` rows only (tools/pagefind_bench/prepare.py, doccorner_sources.json)."""
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
sys.path.insert(0, os.path.join(HERE, '..', '..', 'pagefind_bench'))
GH, GW = 121, 177
SKIP = ('book', 'negative', 'midv')   # book pages bend; negatives have no page; MIDV is research-only anyway


def homography_grid(q, rows, cols, turned=False):
    """the page grid through the quad's homography (q: TL TR BR BL photo px): 2 x rows x cols photo px"""
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), np.float32(q))
    if turned:   # the net's grid for a page it sees turned (synth.py grid_turned): row i -> s, column j -> t = 1 - j
        s, t = np.meshgrid(np.linspace(0, 1, rows), 1 - np.linspace(0, 1, cols), indexing='ij')
    else:
        s, t = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    p = cv2.perspectiveTransform(np.stack([s, t], -1).reshape(-1, 1, 2).astype(np.float64), H).reshape(rows, cols, 2)
    return p.transpose(2, 0, 1).astype(np.float32)


def straight_sides(img, q):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    H, W = g.shape
    q = np.asarray(q, np.float64)
    diag = np.hypot(*(q.max(0) - q.min(0)))
    offs, found, tested = [], 0, 0
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        r = 0.025 * L
        for t in np.linspace(0.08, 0.92, 24):
            c = a + t * (b - a)
            if not (4 <= c[0] < W - 4 and 4 <= c[1] < H - 4):
                continue
            tested += 1
            o = np.linspace(-r, r, max(9, int(2 * r)))
            pts = c[None] + o[:, None] * n[None]
            ok = (pts[:, 0] >= 1) & (pts[:, 0] < W - 1) & (pts[:, 1] >= 1) & (pts[:, 1] < H - 1)
            if ok.sum() < 5:
                continue
            xs, ys = pts[ok, 0].astype(np.float32), pts[ok, 1].astype(np.float32)
            gxs = cv2.remap(gx, xs[:, None], ys[:, None], cv2.INTER_LINEAR).ravel()
            gys = cv2.remap(gy, xs[:, None], ys[:, None], cv2.INTER_LINEAR).ravel()
            mag = np.abs(gxs * n[0] + gys * n[1])
            if mag.max() < 12:
                continue
            found += 1
            offs.append(abs(o[ok][np.argmax(mag)]) / diag)
    if tested < 12 or found < 0.7 * tested:
        return False, dict(tested=tested, found=found)
    offs = np.array(offs)
    good = np.median(offs) <= 0.003 and np.percentile(offs, 90) <= 0.008
    return bool(good), dict(tested=tested, found=found, med=round(float(np.median(offs)), 4), p90=round(float(np.percentile(offs, 90)), 4))


def init(pass1):
    cv2.setNumThreads(1)
    import make_data
    make_data.init(pass1)


def work(job):
    import make_data
    import synth
    from prepare import read_photo
    rows_in, out, shard_id = job
    rng = np.random.default_rng(shard_id)
    shard = os.path.join(out, f's{shard_id:06d}')
    os.makedirs(shard, exist_ok=True)
    rows, stats = [], {'kept': 0, 'bent': 0, 'crop': 0, 'unread': 0}
    for r in rows_in:
        img = read_photo(r['photo'])
        if img is None:
            stats['unread'] += 1
            continue
        q = np.array(r['corners'], np.float64)
        ok, info = straight_sides(img, q)
        if not ok:
            stats['bent'] += 1
            continue
        sm = dict(photo=img, grid=homography_grid(q, GH, GW), grid_turned=homography_grid(q, GH, GW, turned=True))
        u = rng.random()
        if make_data.PASS1 is not None and u < 0.3:
            crop, ai = 'pass1', synth.app_input(sm, rng, margin=(0.06, 0.06), jitter=0, corners=make_data.pass1_corners(img))
        elif u < 0.5:
            crop, ai = 'wide', synth.app_input(sm, rng, margin=(0.06, 0.16), jitter=0.05)
        else:
            crop, ai = 'gt', synth.app_input(sm, rng)
        if ai is None:
            stats['crop'] += 1
            continue
        x, t, turned = ai
        if not np.isfinite(t).all() or np.abs(t).max() > 1:
            stats['crop'] += 1
            continue
        sid = f'{shard_id:06d}_{len(rows):04d}'
        cv2.imwrite(os.path.join(shard, sid + '.jpg'), cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
        np.save(os.path.join(shard, sid + '.npy'), t.astype(np.float32))
        rows.append({'id': sid, 'path': f's{shard_id:06d}/{sid}', 'kind': 'flat', 'aspect': 'real', 'hw': 0, 'turned': bool(turned),
                     'crop': crop, 'src': r['id'], 'source': r['source'], 'straight': info})
        stats['kept'] += 1
    man = os.path.join(out, 'manifests', f'{shard_id:06d}.json')
    with open(man + '.tmp', 'w') as f:
        json.dump({'seed': shard_id, 'rows': rows, 'stats': stats}, f)
    os.replace(man + '.tmp', man)
    return stats


if __name__ == '__main__':
    import make_data
    ap = argparse.ArgumentParser()
    ap.add_argument('index'); ap.add_argument('out'); ap.add_argument('--split', default='train')
    ap.add_argument('--max_per_source', type=int, default=2000); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pass1', default=''); ap.add_argument('--per_job', type=int, default=100); ap.add_argument('--seed', type=int, default=700000)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.index)]
    rows = [r for r in rows if r['split'] == a.split and r['licence'] == 'train' and not r['negative'] and r['corners']
            and not any(s in r['source'] for s in SKIP)]
    rng = np.random.default_rng(0)
    by = {}
    for r in rows:
        by.setdefault(r['source'], []).append(r)
    rows = []
    for s, rs in sorted(by.items()):
        rows += [rs[i] for i in rng.permutation(len(rs))[:a.max_per_source]]
    rows.sort(key=lambda r: (r['photo'].rsplit('#', 1)[0], int(r['photo'].rsplit('#', 1)[1]) if '#' in r['photo'] else 0))
    os.makedirs(os.path.join(a.out, 'manifests'), exist_ok=True)
    jobs = [(rows[i:i + a.per_job], a.out, a.seed + k) for k, i in enumerate(range(0, len(rows), a.per_job))
            if not os.path.exists(os.path.join(a.out, 'manifests', f'{a.seed + k:06d}.json'))]
    print(len(rows), 'photos from', len(by), 'sources;', len(jobs), 'jobs', flush=True)
    tot = {}
    with Pool(a.workers, initializer=init, initargs=(a.pass1,)) as pool:
        for k, st in enumerate(pool.imap_unordered(work, jobs)):
            for kk, v in st.items():
                tot[kk] = tot.get(kk, 0) + v
            make_data.write_index(a.out, make_data.read_manifests(a.out))
            print(f'job {k + 1}/{len(jobs)} {tot}', flush=True)
