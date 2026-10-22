"""Synthetic filter-eval pages with ground truth (tools/filters/).

  python -I make_synth.py colour OUTDIR N [SEED]   flat colour pages, lit badly (shadow_net/synth.py content
                                                   and shading: pastel cells, highlighter, photos, stamps, blocks,
                                                   cast shadows, colour casts, blur, noise, JPEG)
  python -I make_synth.py screen OUTDIR N [SEED]   photos of an LCD showing a page: RGB-stripe subpixels with a
                                                   black matrix, camera at a small rotation and a non-integer
                                                   scale (moire), lens blur, Bayer mosaic + demosaic, rolling-
                                                   shutter / PWM bands, glare, emissive white point, noise, JPEG

Each sample: NNN_in.png (what the filters get: the flattened page) and NNN_gt.png (the ideal scan: the page
content with its paper / screen background at pure white, colours as printed / displayed).
"""
import json, os, random, sys
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shadow_net'))
import synth  # noqa: E402  (round-6 ShadeNet data generator: page pool, colour content, shading)


def gt_of(a):
    paper = np.percentile(a.reshape(-1, 3), 90, axis=0)
    return np.clip(a / np.maximum(paper, 1e-3), 0, 1)


def colour_sample(rng, h, w):
    a = synth.page_albedo(rng, h, w)
    if rng.random() < 0.5:          # page_albedo colourises 70%: make it ~85%
        synth.colourise(rng, a)
    L, kinds = synth.shading(rng, h // 2, w // 2)
    L = cv2.resize(L, (w, h), interpolation=cv2.INTER_LINEAR)
    ref = np.percentile(cv2.resize(L, (w // 8, h // 8), interpolation=cv2.INTER_AREA).reshape(-1, 3), 99.5, axis=0)
    T = np.clip(L / ref, 0.02, 1.0)
    cast = np.float32([rng.uniform(0.86, 1.04) for _ in range(3)])
    o = a * T * cast * np.float32(rng.uniform(0.7, 1.1))
    o = cv2.GaussianBlur(o, (0, 0), rng.uniform(0.4, 1.2))
    sig = np.float32(rng.uniform(0.004, 0.025))
    o += np.random.default_rng(rng.randrange(1 << 30)).standard_normal(o.shape, dtype=np.float32) * sig * np.sqrt(np.maximum(o, 0.02))
    o = np.clip(o, 0, 1)
    q = rng.randint(55, 92)
    _, enc = cv2.imencode('.jpg', (o * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), gt_of(a), {'shadows': kinds, 'jpeg_q': q}


def screen_sample(rng, h, w):
    """camera image (h x w) of an LCD showing a page; the page fills the frame (as after flattening)"""
    # screen content at screen resolution: the screen shows the page at sw x sh screen pixels; the
    # camera sees one screen pixel as `ppx` camera pixels (1.05 .. 3: the grid aliases into moire)
    ppx = rng.uniform(0.7, 1.4) if rng.random() < 0.35 else rng.uniform(1.4, 4.5)
    sw, sh = int(w / ppx) + 2, int(h / ppx) + 2
    a = synth.page_albedo(rng, max(sh, 64), max(sw, 64))   # the page, rendered at screen resolution
    if rng.random() < 0.5:
        synth.colourise(rng, a)
    a = a[:sh, :sw]
    gt_screen = gt_of(a)
    # emissive screen: content times the screen's white (blue-ish) plus a black glow / reflection
    white = np.float32([rng.uniform(0.95, 1.0), rng.uniform(0.88, 0.97), rng.uniform(0.78, 0.92)])  # B G R
    if rng.random() < 0.3:
        white = np.float32([rng.uniform(0.85, 0.95), rng.uniform(0.9, 0.97), 1.0])   # warm (night mode)
    black = np.float32(rng.uniform(0.02, 0.12))
    disp = gt_screen * white + black * (1 - gt_screen)
    # subpixel simulation: each screen pixel = K x K sim pixels: R, G, B stripes with a dark gap
    K = 3
    up = np.repeat(np.repeat(disp, K, 0), K, 1)
    order = [2, 1, 0] if rng.random() < 0.85 else [0, 1, 2]   # RGB stripes (BGR channel index), sometimes BGR panels
    sim = np.zeros_like(up)
    for s, ch in enumerate(order):
        sim[:, s::K, ch] = 3.0 * up[:, s::K, ch]   # each stripe is 1/3 of the pixel, 3x as bright
    del up
    if rng.random() < 0.7:   # black matrix between pixel rows
        my = np.ones(K, np.float32); my[K - 1] = rng.uniform(0.0, 0.6)
        sim *= np.tile(my, sh)[:, None, None]
    # camera: sim -> camera pixels with a small rotation, lens blur (pre-filter in sim units), sampling
    scale = ppx / K   # camera px per sim px
    ang = rng.uniform(-2.5, 2.5) if rng.random() < 0.8 else rng.uniform(-8, 8)
    blur_cam = rng.uniform(0.35, 0.9)   # lens + sensor PSF sigma, camera px
    sim = cv2.GaussianBlur(sim, (0, 0), blur_cam / scale)
    M = cv2.getRotationMatrix2D((sim.shape[1] / 2, sim.shape[0] / 2), ang, scale)
    M[0, 2] += w / 2 - sim.shape[1] / 2
    M[1, 2] += h / 2 - sim.shape[0] / 2
    cam = cv2.warpAffine(sim, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    M2 = M.copy()   # the same placement for the content at screen resolution (screen px u -> sim px K u + (K - 1) / 2)
    M2[:, :2] = M[:, :2] * K
    M2[:, 2] = M[:, 2] + M[:, :2] @ np.float64([(K - 1) / 2, (K - 1) / 2])
    gt = cv2.warpAffine(gt_screen, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gt = cv2.GaussianBlur(gt, (0, 0), 0.5)
    # exposure: the camera exposes for the bright screen
    cam *= np.float32(rng.uniform(0.75, 1.0)) / max(1e-3, float(np.percentile(cam, 99)))
    # rolling shutter / PWM bands: horizontal brightness bands, period 40..400 px
    if rng.random() < 0.6:
        yy = np.arange(h, dtype=np.float32)[:, None, None]
        per = rng.uniform(40, 400)
        cam *= 1 + rng.uniform(0.03, 0.15) * np.sin(2 * np.pi * yy / per + rng.uniform(0, 6.3))
    # glare / reflection: a smooth bright blob, slightly warm
    if rng.random() < 0.4:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.1, 0.4) * min(h, w)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(0.1, 0.35)
        cam += g[..., None] * np.float32([0.9, 0.95, 1.0])
    # vignetting of the screen's viewing angle
    cam *= (1 + synth.smooth_field(rng, h, w, 2, 0.06))[..., None]
    cam = np.clip(cam, 0, 1)
    # Bayer mosaic (RGGB) and demosaic, noise before demosaic, JPEG
    raw = np.zeros((h, w), np.float32)
    raw[0::2, 0::2] = cam[0::2, 0::2, 2]
    raw[0::2, 1::2] = cam[0::2, 1::2, 1]
    raw[1::2, 0::2] = cam[1::2, 0::2, 1]
    raw[1::2, 1::2] = cam[1::2, 1::2, 0]
    sig = rng.uniform(0.004, 0.02)
    raw += np.random.default_rng(rng.randrange(1 << 30)).standard_normal(raw.shape, dtype=np.float32) * sig * np.sqrt(np.maximum(raw, 0.02))
    raw8 = np.clip(raw * 255 + 0.5, 0, 255).astype(np.uint8)
    bgr = cv2.demosaicing(raw8, cv2.COLOR_BayerRG2BGR if rng.random() < 0.5 else cv2.COLOR_BayerRG2BGR_EA)
    q = rng.randint(70, 95)
    _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), gt, {'px_per_screen_px': round(ppx, 3), 'angle': round(ang, 2), 'jpeg_q': q}


if __name__ == '__main__':
    kind, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 11
    os.makedirs(out, exist_ok=True)
    meta = {}
    for i in range(n):
        rng = random.Random(seed * 1000 + i)
        if kind == 'colour':
            h, w = (1600, 1200) if rng.random() < 0.8 else (1200, 1600)
            o, g, m = colour_sample(rng, h, w)
        else:
            h, w = (1500, 1100) if rng.random() < 0.6 else (1100, 1500)
            o, g, m = screen_sample(rng, h, w)
        cv2.imwrite(f'{out}/{i:03d}_in.png', o)
        cv2.imwrite(f'{out}/{i:03d}_gt.png', (g * 255 + 0.5).astype(np.uint8))
        meta[f'{i:03d}'] = m
        print(i, m, flush=True)
    json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
