"""Filter evaluation (tools/filters/): renders filters with a desktop build of the core and measures them.

  python -I eval_filters.py render BIN SETDIR OUTDIR [--filters 0,2,5] [--strength 0.7] [--jobs 4] [--extra "..."]
  python -I eval_filters.py score SETDIR OUTDIR [--filters 2,5] [--json OUT.json]
  python -I eval_filters.py compare A.json B.json

A set directory holds either NAME_page.png (+ NAME_paper.png, the paper alpha) - real flattened pages - or
NNN_in.png + NNN_gt.png - synthetic pages with the ideal scan. BIN is tools/desktop_check/build/*/filter_cli.exe.

Metrics per page and filter (all 8-bit, BGR):
  marks_lost   ink marks (8-connected, >= 3 px, >= 20 levels below the 41-px local median of the reference)
               with no pixel >= 8 levels below the output's local median (tools/filter_check.py)
  paper_white  % of paper pixels with every channel >= 250 (paper: GT white and away from ink; real pages:
               estimated - bright, near-neutral after white balance, ink-free 9 px around)
  paper_unev   p95 - p5 of the output luminance blurred (sigma 6) over paper pixels (lower = more even)
  ink_L        median output luma of ink cores (reference darkest ink: >= 60 below the local median)
  sat_ratio    per colour region (ink-free background of saturation >= 0.12, >= 0.15 % of the page):
               saturation (max - min) / max of the region's median colour, output / reference
               (reference: the GT, or the input divided by its paper colour); worst and median reported
  hue_err      chroma-weighted hue difference of the region colours, degrees (median over regions)
  white_regions  colour regions whose output median is near white (min channel >= 245) - holes
  halo         % of pixels within 3 px of ink edges inside colour regions that are >= 20 levels brighter
               than the region median in the output but not in the reference
  bg_resid     (GT only) std of the output luma over GT-white paper pixels after a 3x3 mean: moire, bands, noise
  bg_chroma    (GT only) mean chroma (max - min) of the output over GT-white paper pixels
  dE           (GT only) mean CIE76 dE between output and GT over all pixels; dE_col over colour regions
"""
import glob, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2


def pages(setdir):
    out = []
    for p in sorted(glob.glob(os.path.join(setdir, '*_page.png'))):
        n = os.path.basename(p)[:-9]
        pa = os.path.join(setdir, n + '_paper.png')
        out.append((n, p, pa if os.path.exists(pa) else None, None))
    if out:
        return out
    for p in sorted(glob.glob(os.path.join(setdir, '*_in.png'))) + sorted(glob.glob(os.path.join(setdir, '*_in.jpg'))):
        n = os.path.basename(p).rsplit('_in.', 1)[0]
        g = [q for q in glob.glob(os.path.join(setdir, n + '_gt.*'))]
        out.append((n, p, None, g[0] if g else None))
    return out


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def render(bin_, setdir, outdir, filters, strength, jobs, extra):
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ)
    env['PATH'] = r'C:\Users\User\Android\opencv\build\x64\vc16\bin;' + env['PATH']

    def one(pg):
        n, p, pa, g = pg
        d = os.path.join(outdir, n)
        os.makedirs(d, exist_ok=True)
        p, d = p.replace(os.sep, '/'), d.replace(os.sep, '/')
        cmd = [bin_, p, d, '--filters', filters, '--strength', str(strength)] + (['--paper', pa] if pa else []) + extra.split()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print('FAIL', n, r.stderr[-300:], flush=True)
            return None
        return json.loads(r.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(one, pages(setdir)))
    json.dump([r for r in res if r], open(os.path.join(outdir, 'render.json'), 'w'), indent=1)
    ms = [f['ms'] for r in res if r for f in r['filters']]
    pp = [r['page_params_ms'] for r in res if r]
    print(f'rendered {len(res)} pages; page_params median {np.median(pp):.0f} ms; filter median {np.median(ms):.0f} ms')


# ---------------------------------------------------------------- metrics
def lum(img):
    return (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(np.float32)


def contrast(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(g, 41).astype(np.int16) - g


def ink_free_bg(img, k=15):
    """background colour: text removed by a closing (max filter then min filter), then a median"""
    c = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.medianBlur(c, 9)


def sat(c):
    c = np.asarray(c, np.float32)
    mx = c.max(-1)
    return (mx - c.min(-1)) / np.maximum(mx, 1)


def hue(c):
    c = np.clip(np.asarray(c, np.float32), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(c, cv2.COLOR_BGR2HSV_FULL)[:, 0, 0].astype(np.float32) * 360 / 256


def lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)


class Ref:
    """everything about a page that does not depend on the filter"""

    def __init__(self, inp, gt):
        self.inp, self.gt = inp, gt
        ref = gt if gt is not None else inp
        self.c0 = contrast(ref)
        self.ink = self.c0 >= 20
        n, self.lab_, st, _ = cv2.connectedComponentsWithStats(self.ink.astype(np.uint8), connectivity=8)
        self.n = n
        self.big = st[:, 4] >= 3
        self.big[0] = False
        self.core = self.c0 >= 60
        near_ink = cv2.dilate(self.ink.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        h, w = ref.shape[:2]
        if gt is not None:
            self.paper = (gt.min(-1) >= 250) & ~near_ink
            bgref = ink_free_bg(gt).astype(np.float32)
        else:
            bg = ink_free_bg(inp).astype(np.float32)
            L = lum(bg)
            loc = cv2.dilate(L, np.ones((101, 101), np.uint8))
            cand = L >= 0.8 * loc
            pc = np.median(bg[cand], axis=0) if cand.any() else np.float32([200, 200, 200])
            wbn = bg / np.maximum(pc, 1) * pc.mean()
            self.paper = cand & (sat(wbn) < 0.08) & ~near_ink
            # reference colours: the input divided by its local paper level (paper -> white)
            Lp = cv2.GaussianBlur(np.where(self.paper, L, 0).astype(np.float32), (0, 0), 25)
            Wp = cv2.GaussianBlur(self.paper.astype(np.float32), (0, 0), 25)
            S = np.where(Wp > 0.02, Lp / np.maximum(Wp, 1e-6), np.median(L[self.paper]) if self.paper.any() else 200)
            bgref = np.clip(bg / pc * pc.mean() / np.maximum(S, 1)[..., None] * 255, 0, 255)
        self.bgref = bgref
        # colour regions: ink-free background that is a colour (saturation >= 0.12) and not too dark
        s = sat(bgref)
        reg = ((s >= 0.12) & (bgref.max(-1) >= 40)).astype(np.uint8)
        reg = cv2.morphologyEx(reg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m, rl, rst, _ = cv2.connectedComponentsWithStats(reg, connectivity=8)
        self.regions = [k for k in range(1, m) if rst[k, 4] >= 0.0015 * h * w]
        self.rboxes = [tuple(rst[k, :4]) for k in self.regions]
        self.rlab = rl
        self.Lref = lum(ref)
        self.edges = cv2.dilate((cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 50, 150) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


def ssim(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cab = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    m = ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return float(m.mean())


def score(ref, out):
    r = {}
    c1 = contrast(out)
    kept = c1 >= 8
    keep = np.zeros(ref.n, bool)
    keep[np.unique(ref.lab_[kept & ref.ink])] = True
    r['marks_lost'] = int((ref.big & ~keep).sum())
    r['marks'] = int(ref.big.sum())
    L = lum(out)
    if ref.paper.sum() > 1000:
        r['paper_white'] = round(100.0 * float((out.min(-1)[ref.paper] >= 250).mean()), 2)
        Lb = cv2.GaussianBlur(L, (0, 0), 6)[ref.paper]
        r['paper_unev'] = round(float(np.percentile(Lb, 95) - np.percentile(Lb, 5)), 2)
    if ref.core.sum() > 100:
        r['ink_L'] = round(float(np.median(L[ref.core])), 1)
    sr, he, white, halo_n, halo_d = [], [], 0, 0, 0
    bgo = ink_free_bg(out).astype(np.float32)
    Lr = ref.Lref
    for k, (x, y, w, h) in zip(ref.regions, ref.rboxes):
        m = ref.rlab[y:y + h, x:x + w] == k
        ci = np.median(ref.bgref[y:y + h, x:x + w][m], axis=0)
        co = np.median(bgo[y:y + h, x:x + w][m], axis=0)
        sr.append(float(sat(co) / max(sat(ci), 1e-3)))
        d = abs(float(hue(co)[0] - hue(ci)[0]))
        he.append(min(d, 360 - d))
        if co.min() >= 245:
            white += 1
        e = m & ref.edges[y:y + h, x:x + w]
        if e.sum() > 50:
            Lo, Li = L[y:y + h, x:x + w], Lr[y:y + h, x:x + w]
            mo, mi = float(np.median(Lo[m])), float(np.median(Li[m]))
            halo_n += int(((Lo[e] >= mo + 20) & (Li[e] < mi + 5)).sum())
            halo_d += int(e.sum())
    r['regions'] = len(ref.regions)
    if sr:
        r['sat_worst'] = round(min(sr), 3)
        r['sat_med'] = round(float(np.median(sr)), 3)
        r['hue_err'] = round(float(np.median(he)), 1)
        r['white_regions'] = white
        r['halo'] = round(100.0 * halo_n / max(1, halo_d), 2)
    if ref.gt is not None:
        g = ref.gt
        wm = (g.min(-1) >= 250) & ~(cv2.dilate(ref.ink.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        if wm.sum() > 1000:
            Lm = cv2.blur(L, (3, 3))
            r['bg_resid'] = round(float(Lm[wm].std()), 2)
            r['bg_chroma'] = round(float((out.max(-1).astype(np.float32) - out.min(-1))[wm].mean()), 2)
        # tone-invariant fidelity: per-channel gain + offset fitted from the output to the GT, then
        # PSNR (all channels) and SSIM (luma); a shadow left behind or moire stays, a different
        # paper level or contrast does not count
        of, gf = out.reshape(-1, 3).astype(np.float64), g.reshape(-1, 3).astype(np.float64)
        fit = np.empty_like(of)
        for c in range(3):
            A = np.stack([of[:, c], np.ones(len(of))], 1)
            k, *_ = np.linalg.lstsq(A[::7], gf[::7, c], rcond=None)
            fit[:, c] = A @ k
        fit = np.clip(fit, 0, 255)
        mse = float(((fit - gf) ** 2).mean())
        r['psnr_fit'] = round(10 * np.log10(255 ** 2 / max(mse, 1e-6)), 2)
        r['ssim_fit'] = round(ssim(lum(fit.reshape(out.shape)), lum(g)), 4)
        la, lb = lab(out), lab(g)
        dE = np.sqrt(((la - lb) ** 2).sum(-1))
        r['dE'] = round(float(dE.mean()), 2)
        if ref.regions:
            m = np.isin(ref.rlab, ref.regions)
            r['dE_col'] = round(float(dE[m].mean()), 2)
    return r


def score_page(job):
    n, p, g, outdir, filters = job
    cv2.setNumThreads(1)
    inp = cv2.imread(p)
    gt = cv2.imread(g) if g else None
    ref = Ref(inp, gt)
    r = {}
    for k in filters:
        f = os.path.join(outdir, n, f'f{k}.png')
        if os.path.exists(f):
            r[k] = score(ref, cv2.imread(f))
    return n, r


def score_set(setdir, outdir, filters, jobs=4):
    from multiprocessing import Pool
    js = [(n, p, g, outdir, filters) for n, p, pa, g in pages(setdir)]
    with Pool(jobs) as pool:
        return dict(pool.map(score_page, js))


def summary(res):
    s = {}
    for n, fr in res.items():
        for k, r in fr.items():
            for m, v in r.items():
                s.setdefault(k, {}).setdefault(m, []).append(v)
    out = {}
    for k, ms in s.items():
        o = {}
        for m, v in ms.items():
            v = np.float64(v)
            if m in ('marks_lost', 'white_regions', 'marks', 'regions'):
                o[m + '_sum'] = int(v.sum())
                o[m + '_max'] = int(v.max())
            elif m == 'sat_worst':
                o['sat_worst'] = round(float(v.min()), 3)
                o['sat_worst_p10'] = round(float(np.percentile(v, 10)), 3)
            else:
                o[m] = round(float(np.median(v)), 2)
                if m in ('paper_white',):
                    o[m + '_p10'] = round(float(np.percentile(v, 10)), 2)
        out[k] = o
    return out


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'render':
        render(sys.argv[2], sys.argv[3], sys.argv[4], arg('--filters', '0,2,5'), float(arg('--strength', 0.7)), int(arg('--jobs', 4)),
               arg('--extra', ''))
    elif cmd == 'score':
        filters = arg('--filters', '2,5').split(',')
        res = score_set(sys.argv[2], sys.argv[3], filters)
        summ = summary(res)
        js = arg('--json', os.path.join(sys.argv[3], 'scores.json'))
        json.dump({'pages': res, 'summary': summ}, open(js, 'w'), indent=1)
        for k, o in summ.items():
            print(f'filter {k}:', json.dumps(o))
    elif cmd == 'compare':
        a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
        for k in a['summary']:
            if k in b['summary']:
                print(f'filter {k}')
                for m in a['summary'][k]:
                    print(f'   {m:18s} {a["summary"][k][m]!s:>10} -> {b["summary"][k].get(m)!s:>10}')
