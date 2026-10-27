"""synth.py: one training photo for the flattening nets with its exact ground truth (geom.py), and the app's crop.

sample(rng, pages, kind=None) -> dict
  photo   H x W x 3 uint8 BGR (a phone-like photo: 1600 x 1200 or 1200 x 1600 ...)
  grid    2 x GH x GW float32: photo px of each page grid point (row i: t = i / (GH - 1), col j: s = j / (GW - 1))
  kind    the paper's shape ('flat', 'nearflat', 'curl', 'corner', 'fold', 'fold2', 'curl_corner', 'wave_curl')
  hw      the paper's true height / width
pages: a list of flat page image paths (textures.py makes them); a page's aspect sets the paper's aspect, except
that narrow objects (receipts, tickets, bills, cards) are cut from pages with random aspects (see ASPECTS).

app_input(sample, rng, size=(1024, 1440), margin=(0.02, 0.10), jitter=0.02) -> (x uint8 HxWx3 RGB, target
2 x GH x GW in [-1, 1] of x, turned): as the app makes SD-HR's input (core/src/uvdoc.cpp uv_crop / uv_input on pass 1's
corners): the page's box from its corners (jittered: pass 1 is not exact) + a margin of the long side, clamped to
the photo, turned 90 deg clockwise when the page lies wider than tall, INTER_AREA to size.
"""
import math
import os

import cv2
import numpy as np

import geom

GH, GW = 121, 177          # SD-HR's grid (rows x cols)
ASPECTS = [  # (name, h/w range, weight): the papers people scan
    ('a4', (1.38, 1.44), 0.30), ('letter', (1.27, 1.31), 0.10), ('receipt', (1.8, 5.0), 0.16),
    ('bill', (0.40, 0.48), 0.06), ('card', (0.55, 0.66), 0.06), ('coupon_strip', (2.0, 3.5), 0.05),
    ('square', (0.9, 1.1), 0.05), ('landscape', (0.62, 0.78), 0.12), ('book', (1.45, 1.6), 0.10)]
_TEX = {}
REJECT = {}   # why samples were left out (counts)
# share of samples whose paper runs out of the photo (up to 30 % of its grid outside): photos taken close up, where
# the photo's edge cuts the paper. Without them the net reads a frame edge across the paper as a curled corner
# (round 17: the first fine-tune bent a flat table page whose sheet ran out of the photo)
CUT_P = float(os.environ.get('FS_CUT_P', 0))   # off by default: tried 2026-10-26, no evidence it helps; targets beyond the photo edge are unknowable
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'


_GRIDS = {}


def pixel_grid(W, H):
    """(yy, xx) float32 H x W pixel coordinates, made once per photo size (np.mgrid each time cost ~0.1 s a sample)"""
    if (W, H) not in _GRIDS:
        yy, xx = np.mgrid[0:H, 0:W]
        _GRIDS[(W, H)] = (yy.astype(np.float32), xx.astype(np.float32))
    return _GRIDS[(W, H)]


def noise(rng, shape, sigma):
    """Gaussian noise in float32 (float64 noise over a whole photo was a large share of a sample's time)"""
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(sigma)


def load_page(path):
    if path not in _TEX:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if len(_TEX) > 24:   # 6000 pages drawn at random: a big cache rarely hits and costs GBs per worker
            _TEX.pop(next(iter(_TEX)))
        _TEX[path] = im
    return _TEX[path]


def pick_aspect(rng):
    w = np.array([a[2] for a in ASPECTS])
    name, (lo, hi), _ = ASPECTS[rng.choice(len(ASPECTS), p=w / w.sum())]
    return name, rng.uniform(lo, hi)


def page_texture(rng, pages, hw, long_px=1800):
    """a page image of aspect hw (height / width): a random page cut to that aspect, at most long_px on its long side"""
    im = load_page(pages[rng.integers(len(pages))])
    h, w = im.shape[:2]
    # the largest window of aspect hw inside a random part of the page
    full = rng.random() < 0.5
    if h / w > hw:
        cw = int(w * (1.0 if full else rng.uniform(0.6, 1.0)))
        ch = min(h, int(cw * hw))
    else:
        ch = int(h * (1.0 if full else rng.uniform(0.6, 1.0)))
        cw = min(w, int(ch / hw))
    x0 = rng.integers(0, w - cw + 1)
    y0 = rng.integers(0, h - ch + 1)
    tex = im[y0:y0 + ch, x0:x0 + cw]
    s = long_px / max(tex.shape[:2])
    if s < 1:
        tex = cv2.resize(tex, (max(8, int(tex.shape[1] * s)), max(8, int(tex.shape[0] * s))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.25:   # tinted paper
        tint = np.array([rng.uniform(0.85, 1.0) for _ in range(3)])
        tex = np.clip(tex * tint, 0, 255).astype(np.uint8)
    return tex


def background(rng, size, pages):
    """a table, cloth, another document, a dark desk, a white desk ..., with other papers and objects lying on it
    (round 18: the fine-tune took a cluttered scene for the page and drew background strips along pale edges)"""
    W, H = size
    kind = rng.choice(['solid', 'wood', 'noise', 'page', 'dark', 'pale'], p=[0.18, 0.15, 0.17, 0.10, 0.10, 0.30])
    base = np.array([rng.uniform(40, 230) for _ in range(3)])
    if kind == 'dark':
        base = base * 0.3
    elif kind == 'pale':   # a white or light grey desk: hardly darker than the paper
        base = np.full(3, rng.uniform(195, 250)) + rng.uniform(-4, 4, 3)
    bg = np.ones((H, W, 3), np.float32) * base.astype(np.float32)
    if kind == 'wood':
        y = np.linspace(0, rng.uniform(20, 80), H)[:, None] + rng.normal(0, 1, (H, 1)).cumsum(0) * 0.05
        g = np.sin(y + np.linspace(0, rng.uniform(1, 6), W)[None]) * rng.uniform(8, 30)
        bg = bg + g[..., None].astype(np.float32)
    elif kind in ('noise', 'pale'):
        n = cv2.resize(rng.normal(0, 1, (H // 16 + 1, W // 16 + 1)), (W, H), interpolation=cv2.INTER_CUBIC)
        bg = bg + (n[..., None] * (rng.uniform(5, 30) if kind == 'noise' else rng.uniform(0, 6))).astype(np.float32)
    elif kind == 'page':
        p = load_page(pages[rng.integers(len(pages))])
        p = cv2.resize(p, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        bg = p * rng.uniform(0.5, 0.9)
    bg = bg + noise(rng, bg.shape, rng.uniform(1, 6))
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 4))):
            bg = clutter(rng, bg, pages)
    return np.clip(bg, 0, 255)


def clutter(rng, bg, pages):
    """one thing lying on the table: a paper (a page, a receipt, a card), or an object (a phone, a box, a pen),
    with its contact shadow. It lies under the sheet being scanned (drawn on the table before the sheet): the
    target stays the sheet. Its edge is blended by coverage (no dark rim)."""
    H, W = bg.shape[:2]
    if rng.random() < 0.65:
        name, hw = pick_aspect(rng)
        if name in ('receipt', 'coupon_strip', 'bill', 'card'):
            obj = page_texture(rng, pages, hw, long_px=900)
        else:
            obj = load_page(pages[rng.integers(len(pages))])
        obj = obj.astype(np.float32) * rng.uniform(0.75, 1.05)
        L = rng.uniform(0.2, 0.75) * max(W, H)
    else:   # a solid object: a phone, a box, a pen
        hw = [rng.uniform(1.8, 2.3), rng.uniform(0.5, 1.5), rng.uniform(8, 15)][int(rng.integers(3))]
        col = np.full(3, rng.uniform(10, 90)) if rng.random() < 0.6 else rng.uniform(30, 230, 3)
        obj = np.ones((int(64 * hw), 64, 3), np.float32) * col.astype(np.float32)
        obj += np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), obj.shape[0]).astype(np.float32)[:, None, None]   # a sheen
        L = rng.uniform(0.1, 0.4) * max(W, H) if hw < 5 else rng.uniform(0.15, 0.3) * max(W, H)
    oh, ow = obj.shape[:2]
    h = L if oh >= ow else L * oh / ow
    w = h * ow / oh
    cx, cy = rng.uniform(-0.1, 1.1) * W, rng.uniform(-0.1, 1.1) * H
    ang = rng.uniform(-math.pi, math.pi)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    quad = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) * 0.5
    quad = quad * (1 + rng.uniform(-0.06, 0.06, (4, 1)))   # a little perspective
    quad = (quad @ R.T + [cx, cy]).astype(np.float32)
    Mx = cv2.getPerspectiveTransform(np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32), quad)
    cov = cv2.warpPerspective(np.ones((oh, ow), np.float32), Mx, (W, H), flags=cv2.INTER_LINEAR)
    col = cv2.warpPerspective(obj, Mx, (W, H), flags=cv2.INTER_LINEAR)   # 0 outside: premultiplied by coverage
    if rng.random() < 0.7:
        sh = cv2.GaussianBlur(cov, (0, 0), rng.uniform(2, 10))
        sh = cv2.warpAffine(sh, np.float32([[1, 0, rng.uniform(-6, 6)], [0, 1, rng.uniform(0, 10)]]), (W, H))
        bg = bg * (1 - rng.uniform(0.1, 0.4) * sh[..., None])
    return col + bg * (1 - cov[..., None])


def shade(rng, normals_img, mask, size):
    """brightness: (on the paper: a light direction on its normals and ambient; everywhere: a soft cast shadow and
    vignetting). Returns (paper, scene): the scene part applies to the table too (round 18: a shadow that stopped
    at the paper's edge outlined the paper)."""
    W, H = size
    L = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 1.0])
    L /= np.linalg.norm(L)
    lam = np.clip((normals_img * L).sum(-1), 0, 1)
    amb = rng.uniform(0.35, 0.75)
    s = amb + (1 - amb) * lam
    scene = np.ones((H, W), np.float32)
    if rng.random() < 0.35:   # a soft shadow (a hand, the phone)
        m = np.zeros((H, W), np.float32)
        pts = (rng.uniform(0, 1, (int(rng.integers(3, 7)), 2)) * [W, H]).astype(np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 1.0)
        k = int(rng.uniform(0.02, 0.12) * max(W, H)) | 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        scene = scene * (1 - m * rng.uniform(0.2, 0.6))
    yy, xx = pixel_grid(W, H)
    v = 1 - np.float32(rng.uniform(0, 0.25) * 4) * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    scene = scene * v
    # the table is lit like flat paper (ambient + the light at its angle), so paper and table differ by albedo
    flat = amb + (1 - amb) * L[2]
    return (s * scene)[..., None].astype(np.float32), (flat * scene)[..., None].astype(np.float32)


def sample(rng, pages, kind=None, size=None):
    if size is None:
        size = (1200, 1600) if rng.random() < 0.75 else (1600, 1200)
    W, H = size
    aname, hw = pick_aspect(rng)
    tex = page_texture(rng, pages, hw)
    kind, deform = geom.random_shape(rng, hw, kind)
    # the mesh: fine enough that a face stays ~8 px in the photo (99th percentile), boxes sized to the largest
    nu = 64
    nv = int(np.clip(round(nu * hw), 16, 256))
    P0, st, F = geom.sheet(hw, nu, nv)
    P = deform(P0.reshape(-1, 3)).reshape(nv, nu, 3)
    # upright within +-12 deg: the net reads the crop's top as the page's top (the orientation stage turns the page
    # afterwards), so a photo taken upside down has no single right target
    cut = rng.random() < CUT_P
    cam = geom.random_camera(rng, P.reshape(-1, 3), size, fill=(0.95, 1.35) if cut else (0.6, 1.05), upright=True)
    V2, z = cam.project(P.reshape(-1, 3))
    if (z <= 0).any():
        return None
    V = V2.reshape(nv, nu, 2)
    du = np.percentile(np.abs(np.diff(V, axis=1)).max(-1), 99)
    dv = np.percentile(np.abs(np.diff(V, axis=0)).max(-1), 99)
    nu2 = int(np.clip(nu * du / 8, nu, 500))
    nv2 = int(np.clip(nv * dv / 8, nv, 900))
    if (nu2, nv2) != (nu, nv):
        P0, st, F = geom.sheet(hw, nu2, nv2)
        P = deform(P0.reshape(-1, 3)).reshape(nv2, nu2, 3)
        V2, z = cam.project(P.reshape(-1, 3))
        nu, nv = nu2, nv2
    tri = V2[F]
    box = int(min(48, np.ceil((tri.max(1) - tri.min(1)).max()) + 2))
    try:
        stimg, face = geom.rasterise(V2, z, F, st.reshape(-1, 2), size, device=DEVICE, max_box=box)
    except ValueError:   # a face too big to rasterise (the mesh could not be made fine enough)
        REJECT['big_face'] = REJECT.get('big_face', 0) + 1
        return None
    mask = face >= 0
    # normals per pixel from the face (sheet frame, z up)
    nrm = geom.face_normals(P.reshape(-1, 3), F)
    nrm[nrm[:, 2] < 0] *= -1
    nimg = np.zeros((H, W, 3), np.float32)
    nimg[..., 2] = 1
    nimg[mask] = nrm[face[mask]]
    # the photo
    th, tw = tex.shape[:2]
    mx = (stimg[..., 0] * (tw - 1)).astype(np.float32)
    my = (stimg[..., 1] * (th - 1)).astype(np.float32)
    mx[~mask] = -1
    my[~mask] = -1
    paper = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)).astype(np.float32)
    s_paper, s_scene = shade(rng, nimg, mask, size)
    lit = paper * s_paper
    bg = background(rng, size, pages)
    if rng.random() < 0.65:   # the sheet's own contact shadow on the table (not always: a pale desk may show none)
        sh = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), rng.uniform(2, 8))
        sh = cv2.warpAffine(sh, np.float32([[1, 0, rng.uniform(-8, 8)], [0, 1, rng.uniform(-3, 12)]]), (W, H))
        bg = bg * (1 - rng.uniform(0.06, 0.3) * sh[..., None])
    bg = bg * s_scene
    # soft edge: coverage and the coverage-weighted paper colour blurred together (blurring the mask alone over
    # black paper drew a dark rim round every page: a cue real photos do not have)
    mf = mask.astype(np.float32)
    m = cv2.GaussianBlur(mf, (3, 3), 0)[..., None]
    img = cv2.GaussianBlur(lit * mf[..., None], (3, 3), 0) + bg * (1 - m)
    if rng.random() < 0.3:   # glare
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.05, 0.25) * max(W, H)
        yy, xx = pixel_grid(W, H)
        img += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / np.float32(2 * r * r)) * np.float32(rng.uniform(20, 90)))[..., None] * m
    img = img * np.float32(rng.uniform(0.85, 1.15)) + noise(rng, img.shape, rng.uniform(0.5, 4))
    img = np.clip(img, 0, 255, out=img).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.6:
        ok, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 95))])
        img = cv2.imdecode(b, cv2.IMREAD_COLOR)
    # ground truth: the page grid on the rendered mesh, through the same camera
    gs, gt = np.meshgrid(np.linspace(0, 1, GW), np.linspace(0, 1, GH))
    g2, _ = cam.project(geom.mesh_points(P, gs, gt).reshape(-1, 3))
    grid = g2.reshape(GH, GW, 2).transpose(2, 0, 1).astype(np.float32)
    gx, gy = np.round(g2[:, 0]).astype(int), np.round(g2[:, 1]).astype(int)
    inside = (gx >= 0) & (gy >= 0) & (gx < W) & (gy < H)
    if inside.mean() < (0.7 if cut else 0.97) or (cut and inside.mean() > 0.97):
        REJECT['outside'] = REJECT.get('outside', 0) + 1
        return None   # too much of the page outside the photo (or a cut sample that is not cut)
    want = np.stack(np.meshgrid(np.linspace(0, 1, GW), np.linspace(0, 1, GH)), -1).reshape(-1, 2)
    interior = np.zeros((GH, GW), bool)
    interior[2:-2, 2:-2] = True
    sel = inside & interior.ravel()   # border points may round to a pixel just off the paper
    got = stimg[gy[sel], gx[sel]]
    bad = ~np.isfinite(got[:, 0]) | (np.abs(got - want[sel]).max(1) > 0.02)
    if bad.mean() > 0.01:
        REJECT['hidden'] = REJECT.get('hidden', 0) + 1
        return None   # the paper hides part of itself (a corner curled over): ambiguous, left out
    # the grid as the net reads a page it saw turned 90 deg clockwise (uvdoc.cpp UvGrid::at, turned): net row i is
    # page s = i / (GH - 1), net column j is page t = 1 - j / (GW - 1)
    si, tj = np.meshgrid(np.linspace(0, 1, GH), 1 - np.linspace(0, 1, GW), indexing='ij')
    g3, _ = cam.project(geom.mesh_points(P, si, tj).reshape(-1, 3))
    grid_t = g3.reshape(GH, GW, 2).transpose(2, 0, 1).astype(np.float32)
    return dict(photo=img, grid=grid, grid_turned=grid_t, kind=kind, aspect=aname, hw=hw, visible=float(mask.mean()), cut=bool(cut))


def app_input(sm, rng, size=(1024, 1440), margin=(0.02, 0.10), jitter=0.02, corners=None):
    """corners (TL TR BR BL photo px): the crop's frame (UVDoc pass 1's corners on this photo, as the app has them);
    None: the page's own corners, jittered"""
    photo, grid = sm['photo'], sm['grid']
    H, W = photo.shape[:2]
    c = np.array([grid[:, 0, 0], grid[:, 0, -1], grid[:, -1, -1], grid[:, -1, 0]])   # TL TR BR BL
    ext = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
    if corners is not None:
        c = np.asarray(corners, np.float64)
        ext = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
    c = c + rng.uniform(-jitter, jitter, c.shape) * ext
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    m = rng.uniform(*margin) * max(x1 - x0, y1 - y0)
    bx0, by0 = max(0, int(math.floor(x0 - m))), max(0, int(math.floor(y0 - m)))
    bx1, by1 = min(W, int(math.ceil(x1 + m)) + 1), min(H, int(math.ceil(y1 + m)) + 1)
    if bx1 - bx0 < 16 or by1 - by0 < 16:
        return None
    turned = np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[3] - c[0])
    crop = photo[by0:by1, bx0:bx1]
    t = (sm['grid_turned'] if turned else grid) - np.array([bx0, by0], np.float32)[:, None, None]
    cw, ch = bx1 - bx0, by1 - by0
    if turned:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        t = np.stack([(ch - 1) - t[1], t[0]])
        cw, ch = ch, cw
    x = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    tgt = np.stack([t[0] / (cw - 1) * 2 - 1, t[1] / (ch - 1) * 2 - 1]).astype(np.float32)
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB), tgt, turned
