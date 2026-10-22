"""End-to-end check of the line stage on synthetic stage-1 pages whose rules are known.

Pages are drawn by synth.py at stage-1 size (1600 x 2263) and warped as the nets leave them (the
training generator's warps, a different seed range); the warped pages go through the app's line stage
for flattened pages (pipeline_cli, FS_SCAN=1 FS_SCAN_FLAT=1) and the known rule masks through the
map it returns. A rule's straightness is the p95 distance of its centre line from its straight fit;
the input pages give the baseline (no line stage).

  python e2e_eval.py PIPELINE_CLI OUTDIR N [SEED0] [ENV=VAL ...]
    e.g. FS_RULE_NET=rulenet.onnx FS_PYTHON=python3 to run the rule net
Prints per page and in total: rules, median / p90 of the p95 deviation (px), share above 2 px, for
the input and for the output; and how often the line stage was rejected.
Round 11: RN_E2E_GEN=synth2 draws the harder pages of synth2.py (show-through, sheets behind, headings,
barcodes, handwriting, creases, pictures, paper and camera); RN_E2E_JOBS pages at once (default 1)."""
import json, os, random, subprocess, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth

cli, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 50000
extra = dict(a.split('=', 1) for a in sys.argv[5:])
os.makedirs(out, exist_ok=True)
W, H = 1600, 2263
GEN = os.environ.get('RN_E2E_GEN', 'synth')
JOBS = int(os.environ.get('RN_E2E_JOBS', 1))


def page(seed):
    """the warped page (uint8 BGR) and its rule masks (uint8, H and V), cached in OUT"""
    f = os.path.join(out, f'page{seed}.png')
    if os.path.exists(f):
        z = np.load(os.path.join(out, f'page{seed}_gt.npz'))
        return f, z['mh'], z['mv']
    rng = random.Random(seed)
    if GEN == 'synth2':
        import synth2
        pg = synth2.layout2(rng, H, W)
    else:
        pg = synth.layout(rng, H, W)
    mx, my = synth.warp_field(rng, H, W)
    img = cv2.remap(pg.ink, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    mh = cv2.remap(pg.mh, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mv = cv2.remap(pg.mv, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if GEN == 'synth2':
        img = synth2.paper_texture(rng, img)
        if rng.random() < 0.15:
            m2h, m2v = np.dstack([mh, mh]), np.dstack([mv, mv])
            synth2.sheet_behind(random.Random(rng.randrange(1 << 30)), img, m2h, m2v)
            mh, mv = np.ascontiguousarray(m2h[..., 0]), np.ascontiguousarray(m2v[..., 0])
        img = np.clip(synth2.camera(rng, synth.degrade(rng, img)), 0, 1)
    else:
        img = np.clip(synth.degrade(rng, img), 0, 1)
    ok, enc = cv2.imencode('.jpg', (img * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
    cv2.imwrite(f, cv2.imdecode(enc, cv2.IMREAD_COLOR))
    np.savez_compressed(os.path.join(out, f'page{seed}_gt.npz'), mh=mh, mv=mv)
    return f, mh, mv


def straightness(m, hz):
    """p95 deviation from the straight fit of each rule (>= 150 px along) in mask m"""
    A = m if hz else m.T
    n, lab, st, _ = cv2.connectedComponentsWithStats((A > 127).astype(np.uint8), connectivity=8)
    res = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 150 or h > 0.5 * w:
            continue
        sub = lab[y:y + h, x:x + w] == i
        cnt = sub.sum(0)
        cols = np.nonzero(cnt)[0]
        if len(cols) < 100:
            continue
        cy = (sub * np.arange(h)[:, None]).sum(0)[cols] / cnt[cols]
        p = np.polyfit(cols, cy, 1)
        res.append(float(np.percentile(np.abs(cy - np.polyval(p, cols)), 95)))
    return res


def summary(v):
    v = np.array(v) if len(v) else np.zeros(1)
    return f'{len(v):4d} rules  median {np.median(v):.2f}  p90 {np.percentile(v, 90):.2f}  >2px {np.mean(v > 2) * 100:4.1f}%'


def run(seed):
    f, mh, mv = page(seed)
    tag = extra.get('TAG', 'run')
    env = dict(os.environ, FS_SCAN='1', FS_SCAN_FLAT='1', FS_DUMP_MAP=os.path.join(out, f'map{seed}_{tag}.npz'),
               **{k: v for k, v in extra.items() if k != 'TAG'})
    t = time.time()
    r = subprocess.run([cli, f, '-'], env=env, capture_output=True, text=True)
    dt = time.time() - t
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1].replace(chr(92), '/'))
    except Exception:
        return seed, None, r.stderr[-300:], dt, [], []
    z = np.load(env['FS_DUMP_MAP'])
    mx = cv2.resize(z['x'], (z['x'].shape[1] * 4, z['x'].shape[0] * 4), interpolation=cv2.INTER_LINEAR)[:H, :W]
    my = cv2.resize(z['y'], (z['y'].shape[1] * 4, z['y'].shape[0] * 4), interpolation=cv2.INTER_LINEAR)[:H, :W]
    oh = cv2.remap(mh, mx, my, cv2.INTER_LINEAR); ov = cv2.remap(mv, mx, my, cv2.INTER_LINEAR)
    si = straightness(mh, True) + straightness(mv, False)
    so = straightness(oh, True) + straightness(ov, False)
    return seed, j, '', dt, si, so


if __name__ == '__main__':
    for k in range(n):   # pages first (cached), one process
        page(seed0 + k)
    tot_in, tot_out, rejected, nets, t_all = [], [], 0, 0, 0.0
    if JOBS > 1:
        from multiprocessing import Pool
        with Pool(JOBS) as p:
            res = p.map(run, [seed0 + k for k in range(n)])
    else:
        res = [run(seed0 + k) for k in range(n)]
    per = {}
    for seed, j, err, dt, si, so in res:
        if j is None:
            print(seed, 'failed', err); continue
        t_all += dt
        tot_in += si; tot_out += so
        rejected += bool(j.get('lines_rejected')); nets += bool(j.get('rule_net'))
        per[seed] = {'in': si, 'out': so, 'rejected': bool(j.get('lines_rejected')), 'net': j.get('net')}
        print(f"{seed}  in {summary(si)}  |  out {summary(so)}  dist {j.get('line_distortion', -1):.3f}{' REJ' if j.get('lines_rejected') else ''}"
              f"{' spots' if j.get('spots_dropped') else ''}  net {j.get('net')}", flush=True)
    json.dump(per, open(os.path.join(out, 'res_%s.json' % extra.get('TAG', 'run')), 'w'))
    print('TOTAL in ', summary(tot_in))
    print('TOTAL out', summary(tot_out))
    print(f'rejected {rejected}/{n}, rule net ran {nets}/{n}, {t_all / max(1, n):.1f} s/page')
