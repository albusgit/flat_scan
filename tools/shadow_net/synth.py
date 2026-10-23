"""Synthetic training pairs for the shadow net: (photo of a flat page, its shading map).

A sample is a flat page (the albedo: round-4 doc_gen pages plus colour content added here - pastel
cells, highlighter, saturated / grey / dark blocks, gradients, photos, stamps) times a shading map
(smooth light falloff, curl and gutter bands, and hard cast shadows - hands, heads, phones, straight
edges - with a penumbra that varies along the edge), times a colour cast and exposure, then camera
blur, noise and JPEG. The target is the shading map divided by its lit level per channel, so a page
divided by it keeps its lit paper level and colour (paper white balance is the filters' job) and
only the shading is gone. Colour content is never in the target: the net has to learn that blocks,
tints and pictures are the page, and that shadows are not.

  sample(rng, h, w) -> (observed float32 HxWx3 BGR in [0, 1], target float32 HxWx3 in (0, 1], albedo)
  python synth.py N OUT.jpg    contact sheet of N samples (observed | target | albedo)
"""
import glob, os, random, sys
import numpy as np, cv2

POOLS = [r'E:\fs_datasets\shadow_pool', r'E:\fs_datasets\round4\work\pool']
BSDS = sorted(glob.glob(r'E:\fs_datasets\round4\assets\bsds\BSDS500\data\images\*\*.jpg'))
PAGES = sorted(sum((glob.glob(os.path.join(p, '*.jpg')) for p in POOLS), []))


def hsv_bgr(h, s, v):
    """h in [0, 180), s, v in [0, 1] -> BGR float in [0, 1]"""
    return cv2.cvtColor(np.uint8([[[int(h) % 180, int(s * 255), int(v * 255)]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32) / 255


def smooth_field(rng, h, w, cells, amp):
    """low-frequency random field around 0, std ~ amp"""
    g = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 1, (cells, cells)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def rand_rect(rng, h, w, min_frac, max_frac):
    rw = int(w * rng.uniform(min_frac, max_frac)); rh = int(h * rng.uniform(min_frac, max_frac) * rng.uniform(0.3, 1.5))
    rw, rh = max(4, min(w, rw)), max(4, min(h, rh))
    x0 = rng.randint(-rw // 4, w - rw * 3 // 4); y0 = rng.randint(-rh // 4, h - rh * 3 // 4)
    return max(0, x0), max(0, y0), min(w, x0 + rw), min(h, y0 + rh)


def photo(rng, h, w):
    im = cv2.imread(rng.choice(BSDS)).astype(np.float32) / 255
    ih, iw = im.shape[:2]
    s = max(h / ih, w / iw) * rng.uniform(1, 1.8)
    im = cv2.resize(im, (max(w, int(iw * s) + 1), max(h, int(ih * s) + 1)), interpolation=cv2.INTER_AREA)
    y0 = rng.randint(0, im.shape[0] - h); x0 = rng.randint(0, im.shape[1] - w)
    im = im[y0:y0 + h, x0:x0 + w]
    if rng.random() < 0.3:  # dark / low-key pictures
        im = im ** rng.uniform(1.3, 2.5)
    return im


def colourise(rng, a):
    """colour content on the albedo a (float BGR [0, 1], modified in place): what a shadow net must keep"""
    h, w = a.shape[:2]
    paper = np.percentile(a.reshape(-1, 3), 90, axis=0)
    if rng.random() < 0.15:   # slide / poster: chromatic gradient background
        c1 = hsv_bgr(rng.randrange(180), rng.uniform(0.2, 0.9), rng.uniform(0.3, 1))
        c2 = hsv_bgr(rng.randrange(180), rng.uniform(0.2, 0.9), rng.uniform(0.3, 1)) if rng.random() < 0.6 else c1 * rng.uniform(0.5, 1)
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
        t = (t - t.min()) / max(1e-6, t.max() - t.min())
        if rng.random() < 0.4:
            t = np.clip(t * rng.uniform(1, 3) - rng.uniform(0, 1), 0, 1)
        bg = c1 * (1 - t[..., None]) + c2 * t[..., None]
        if rng.random() < 0.5:   # dark slide, light text
            a[:] = 1 - (1 - bg) * (a / paper).clip(0, 1) ** 0.5 if rng.random() < 0.3 else bg * (a / paper).clip(0, 1)
        else:
            a *= bg / paper
    n = rng.choice([0, 0, 0, 0, 1, 1, 2, 3, 5])
    for _ in range(n):   # pastel cells / tinted boxes, crisp edges
        x0, y0, x1, y1 = rand_rect(rng, h, w, 0.08, 0.6)
        c = hsv_bgr(rng.randrange(180), rng.uniform(0.04, 0.45), rng.uniform(0.85, 1))
        a[y0:y1, x0:x1] *= c
    for _ in range(rng.choice([0, 0, 0, 0, 1, 2, 3])):   # solid blocks: saturated, grey, dark; text kept multiplied / inverted
        x0, y0, x1, y1 = rand_rect(rng, h, w, 0.05, 0.5)
        r = rng.random()
        if r < 0.45:
            c = hsv_bgr(rng.randrange(180), rng.uniform(0.4, 1), rng.uniform(0.25, 1))
        elif r < 0.8:
            c = np.full(3, rng.uniform(0.25, 0.9), np.float32) * (1 + np.float32(rng.uniform(-0.03, 0.03)))
        else:
            c = np.full(3, rng.uniform(0.02, 0.2), np.float32)
        reg = (a[y0:y1, x0:x1] / paper).clip(0, 1)
        if c.mean() < 0.45 and rng.random() < 0.6:
            a[y0:y1, x0:x1] = c + (1 - c) * (1 - reg) * rng.uniform(0.6, 1)
        else:
            a[y0:y1, x0:x1] = c * reg
    for _ in range(rng.choice([0, 0, 0, 1, 1, 2, 3, 5])):
        # solid shapes: circles, ellipses (often cut by the page edge, like a cast shadow), blobs,
        # polygons - near-black, grey or coloured, crisp edges. A shadow is never this dark and never
        # hides what is under it: these are the page.
        m = np.zeros((h, w), np.uint8)
        S_ = min(h, w)
        cx = rng.choice([rng.uniform(0, w), rng.uniform(-0.1, 0.1) * w, rng.uniform(0.9, 1.1) * w])
        cy = rng.uniform(-0.1, 1.1) * h
        kind = rng.random()
        if kind < 0.55:
            a_ = S_ * rng.uniform(0.06, 0.4)
            cv2.ellipse(m, (int(cx), int(cy)), (int(a_), int(a_ * rng.uniform(0.5, 1.3))), rng.uniform(0, 180), 0, 360, 1, -1, cv2.LINE_AA)
        elif kind < 0.8:
            k = rng.randint(5, 10); r0 = S_ * rng.uniform(0.06, 0.35)
            ang = np.sort(np.array([rng.uniform(0, 2 * np.pi) for _ in range(k)]))
            rr = r0 * np.array([rng.uniform(0.6, 1.2) for _ in range(k)])
            pts = np.stack([cx + np.cos(ang) * rr, cy + np.sin(ang) * rr], 1).astype(np.int32)
            cv2.fillPoly(m, [pts], 1, cv2.LINE_AA)
            if rng.random() < 0.6:   # smooth blob
                m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), r0 * 0.25) > 0.5).astype(np.uint8)
        else:   # rounded rectangle, rotated
            box = cv2.boxPoints(((cx, cy), (S_ * rng.uniform(0.1, 0.6), S_ * rng.uniform(0.05, 0.3)), rng.uniform(-30, 30))).astype(np.int32)
            cv2.fillPoly(m, [box], 1, cv2.LINE_AA)
        r = rng.random()
        if r < 0.35:
            c = np.float32(hsv_bgr(rng.randrange(180), rng.uniform(0, 0.7), rng.uniform(0.03, 0.25)))   # near black / navy
        elif r < 0.6:
            c = np.full(3, rng.uniform(0.2, 0.85), np.float32) * (1 + np.float32([rng.uniform(-0.04, 0.04) for _ in range(3)]))
        else:
            c = hsv_bgr(rng.randrange(180), rng.uniform(0.15, 1), rng.uniform(0.3, 1))
        mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.5)[..., None]
        reg = (a / paper).clip(0, 1)
        fill = c * reg if rng.random() < 0.5 else np.broadcast_to(c, a.shape)   # text over it kept, or opaque
        a[:] = a * (1 - mf) + fill * mf
    for _ in range(rng.choice([0, 0, 0, 1, 2, 4])):   # highlighter over text rows
        c = rng.choice([(0.45, 1, 1), (0.5, 1, 0.6), (0.85, 0.6, 1), (1, 0.85, 0.5), (0.4, 0.85, 1), (0.75, 1, 0.75)])
        c = np.float32(c) ** rng.uniform(0.6, 1.4)
        hh = int(h * rng.uniform(0.012, 0.035)); ww = int(w * rng.uniform(0.15, 0.8))
        x0 = rng.randint(0, max(1, w - ww)); y0 = rng.randint(0, max(1, h - hh))
        m = np.zeros((h, w), np.float32)
        pts = np.array([[x0, y0], [x0 + ww, y0 + rng.randint(-3, 3)], [x0 + ww + rng.randint(-4, 4), y0 + hh], [x0 + rng.randint(-4, 4), y0 + hh]], np.int32)
        cv2.fillPoly(m, [pts], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), 0.7)[..., None]
        a *= 1 - m + m * c
    if rng.random() < 0.25:   # a picture
        x0, y0, x1, y1 = rand_rect(rng, h, w, 0.2, 0.8)
        if x1 - x0 > 8 and y1 - y0 > 8:
            a[y0:y1, x0:x1] = photo(rng, y1 - y0, x1 - x0)
    for _ in range(rng.choice([0, 0, 0, 0, 1, 2])):   # stamps, coloured ink marks, circles
        c = hsv_bgr(rng.choice([0, 110, 120, 60, 150]) + rng.randint(-8, 8), rng.uniform(0.4, 1), rng.uniform(0.3, 0.8))
        m = np.zeros((h, w), np.float32)
        cx, cy = rng.randrange(w), rng.randrange(h); r = int(min(h, w) * rng.uniform(0.04, 0.15))
        cv2.ellipse(m, (cx, cy), (r, int(r * rng.uniform(0.6, 1))), rng.uniform(0, 180), 0, 360, 1.0, max(1, r // rng.randint(6, 15)))
        if rng.random() < 0.5:
            cv2.ellipse(m, (cx, cy), (r * 6 // 10, int(r * 0.6 * rng.uniform(0.6, 1))), 0, 0, 360, 1.0, -1 if rng.random() < 0.3 else 2)
        m = (cv2.GaussianBlur(m, (0, 0), 0.6) * rng.uniform(0.5, 1))[..., None]
        a *= 1 - m + m * c
    if rng.random() < 0.12:   # dark background slivers at the page edge (left after flattening)
        side = rng.randrange(4); t = int((h if side < 2 else w) * rng.uniform(0.005, 0.03))
        c = np.float32([rng.uniform(0, 0.5)] * 3) * (1 + np.float32([rng.uniform(-0.2, 0.2) for _ in range(3)]))
        if side == 0: a[:t] = c
        elif side == 1: a[-t:] = c
        elif side == 2: a[:, :t] = c
        else: a[:, -t:] = c
    np.clip(a, 0, 1, out=a)


# ---------------------------------------------------------------- shading
_grids = {}


def grids(h, w):
    """cached float32 row / column coordinate grids"""
    if (h, w) not in _grids:
        _grids[(h, w)] = np.mgrid[0:h, 0:w].astype(np.float32)
    return _grids[(h, w)]


def gblur(m, s):
    """Gaussian blur; large sigmas at reduced size (the result is smooth anyway)"""
    if s <= 4:
        return cv2.GaussianBlur(m, (0, 0), max(0.3, s))
    k = int(s / 2)
    h, w = m.shape[:2]
    small = cv2.resize(m, (max(2, w // k), max(2, h // k)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), s / k), (w, h), interpolation=cv2.INTER_LINEAR)


def occluder(rng, h, w):
    """a cast shadow's silhouette (float mask, hard), drawn on a canvas twice the page size so it
    can come in from outside"""
    H, W = 2 * h, 2 * w
    m = np.zeros((H, W), np.uint8)
    ox, oy = w // 2, h // 2
    kind = rng.choices(['hand', 'blob', 'edge', 'object', 'stripe', 'poly', 'foliage'], [0.28, 0.15, 0.18, 0.1, 0.07, 0.08, 0.14])[0]
    S = min(h, w)
    def border_pt():
        side = rng.randrange(4); t = rng.random()
        return [(ox + t * w, oy - 0.1 * h), (ox + t * w, oy + 1.1 * h), (ox - 0.1 * w, oy + t * h), (ox + 1.1 * w, oy + t * h)][side]
    if kind == 'hand':
        # arm from the border, palm, fingers (capsules) spread towards the page
        bx, by = border_pt()
        tx, ty = ox + rng.uniform(0.15, 0.85) * w, oy + rng.uniform(0.15, 0.85) * h
        d = np.array([tx - bx, ty - by]); L = np.linalg.norm(d) + 1e-6; d /= L
        reach = rng.uniform(0.25, 0.9) * L
        px, py = bx + d[0] * reach, by + d[1] * reach
        arm_w = int(S * rng.uniform(0.12, 0.3))
        far = (int(bx - d[0] * S), int(by - d[1] * S))
        cv2.line(m, far, (int(px), int(py)), 1, arm_w)
        pw = int(arm_w * rng.uniform(0.55, 0.8)); ang = np.degrees(np.arctan2(d[1], d[0]))
        cv2.ellipse(m, (int(px), int(py)), (int(pw * 1.1), pw), ang, 0, 360, 1, -1)
        nf = rng.choice([1, 2, 3, 4, 4, 5])
        spread = rng.uniform(0.3, 1.2)
        for i in range(nf):
            fa = np.arctan2(d[1], d[0]) + (i - (nf - 1) / 2) * spread / max(1, nf - 1) + rng.uniform(-0.15, 0.15)
            fl = S * rng.uniform(0.08, 0.28); fw = max(2, int(pw * rng.uniform(0.25, 0.4)))
            sx, sy = px + np.cos(fa) * pw * 0.7, py + np.sin(fa) * pw * 0.7
            ex, ey = sx + np.cos(fa) * fl, sy + np.sin(fa) * fl
            cv2.line(m, (int(sx), int(sy)), (int(ex), int(ey)), 1, fw)
            cv2.circle(m, (int(ex), int(ey)), fw // 2, 1, -1)
        if rng.random() < 0.5:   # thumb
            fa = np.arctan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(1.0, 1.6)
            fl = S * rng.uniform(0.07, 0.2); fw = max(2, int(pw * 0.4))
            cv2.line(m, (int(px), int(py)), (int(px + np.cos(fa) * (pw + fl)), int(py + np.sin(fa) * (pw + fl))), 1, fw)
    elif kind == 'blob':   # head, body, camera: a large ellipse from a border
        bx, by = border_pt()
        ax_ = int(S * rng.uniform(0.2, 0.8)); ay_ = int(ax_ * rng.uniform(0.5, 1.4))
        cv2.ellipse(m, (int(bx), int(by)), (ax_, ay_), rng.uniform(0, 180), 0, 360, 1, -1)
    elif kind == 'edge':   # half plane: a book, a lamp shade, a wall
        ang = rng.uniform(0, 2 * np.pi); n = np.array([np.cos(ang), np.sin(ang)])
        c = np.array([ox + rng.uniform(0.1, 0.9) * w, oy + rng.uniform(0.1, 0.9) * h])
        yy, xx = grids(h, w)
        X, Y = xx + (ox - c[0]), yy + (oy - c[1])   # page pixels in canvas coordinates, from c
        side = X * n[0] + Y * n[1]
        if rng.random() < 0.3:   # slightly curved edge
            side = side + (X * -n[1] + Y * n[0]) ** 2 * rng.uniform(-1, 1) / (3 * S)
        return (side > 0).astype(np.float32), kind
    elif kind == 'object':   # phone (rounded rectangle) or pen
        cx, cy = ox + rng.uniform(0, 1) * w, oy + rng.uniform(0, 1) * h
        if rng.random() < 0.6:
            ww, hh = S * rng.uniform(0.2, 0.5), S * rng.uniform(0.35, 0.9)
        else:
            ww, hh = S * rng.uniform(0.015, 0.04), S * rng.uniform(0.4, 1.2)
        box = cv2.boxPoints(((cx, cy), (ww, hh), rng.uniform(0, 180))).astype(np.int32)
        cv2.fillPoly(m, [box], 1)
        r = max(1, int(min(ww, hh) * 0.15)); m = cv2.dilate(cv2.erode(m, np.ones((r, r), np.uint8)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    elif kind == 'stripe':
        ang = rng.uniform(0, np.pi); n = np.array([np.cos(ang), np.sin(ang)])
        c = np.array([ox + rng.uniform(0, 1) * w, oy + rng.uniform(0, 1) * h])
        yy, xx = grids(h, w)
        dist = np.abs((xx + (ox - c[0])) * n[0] + (yy + (oy - c[1])) * n[1])
        return (dist < S * rng.uniform(0.02, 0.15)).astype(np.float32), kind
    elif kind == 'foliage':   # dappled light through leaves / a plant: clusters of small leaf blobs, holes of light
        cx, cy = ox + rng.uniform(0, 1) * w, oy + rng.uniform(0, 1) * h
        R = S * rng.uniform(0.2, 0.7)
        for _ in range(rng.randint(40, 160)):
            px_, py_ = cx + rng.gauss(0, R * 0.45), cy + rng.gauss(0, R * 0.45)
            a_ = S * rng.uniform(0.015, 0.07)
            cv2.ellipse(m, (int(px_), int(py_)), (int(a_), int(a_ * rng.uniform(0.3, 0.8))), rng.uniform(0, 180), 0, 360, 1, -1)
        if rng.random() < 0.6:   # stems
            for _ in range(rng.randint(1, 5)):
                cv2.line(m, (int(cx + rng.gauss(0, R)), int(cy + rng.gauss(0, R))), (int(cx + rng.gauss(0, R)), int(cy + rng.gauss(0, R))), 1, max(1, int(S * 0.006)))
    else:   # random star-ish polygon
        cx, cy = ox + rng.uniform(0, 1) * w, oy + rng.uniform(0, 1) * h
        k = rng.randint(5, 12); r0 = S * rng.uniform(0.1, 0.45)
        angs = np.sort(np.array([rng.uniform(0, 2 * np.pi) for _ in range(k)]))
        pts = np.stack([cx + np.cos(angs) * r0 * np.array([rng.uniform(0.4, 1.3) for _ in range(k)]),
                        cy + np.sin(angs) * r0 * np.array([rng.uniform(0.4, 1.3) for _ in range(k)])], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m[oy:oy + h, ox:ox + w].astype(np.float32), kind


def penumbra(rng, m, S):
    """blur a hard mask with a penumbra that widens along a random direction (the far end of an
    occluder is further from the paper)"""
    s0 = S * rng.choice([rng.uniform(0.0005, 0.004), rng.uniform(0.002, 0.012), rng.uniform(0.01, 0.06)])
    s1 = s0 * rng.uniform(1, 6)
    a = gblur(m, s0)
    b = gblur(m, s1)
    h, w = m.shape
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = grids(h, w)
    ca, sa = np.cos(ang) / w, np.sin(ang) / h
    lo = min(0, ca * (w - 1)) + min(0, sa * (h - 1)); hi = max(0, ca * (w - 1)) + max(0, sa * (h - 1))
    t = (xx * np.float32(ca) + yy * np.float32(sa) - np.float32(lo)) * np.float32(1 / max(1e-6, hi - lo))
    return a + (b - a) * t


def shading(rng, h, w):
    """the illumination of the page (float BGR, HxWx3, ~(0, 1])"""
    S = min(h, w)
    yy, xx = grids(h, w)
    L = np.ones((h, w, 3), np.float32)
    r = rng.random()
    if r < 0.8:   # smooth falloff: gradient, lamp, vignette
        g = 1 + smooth_field(rng, h, w, rng.choice([2, 3, 4]), rng.uniform(0, 0.15))
        if rng.random() < 0.5:
            ang = rng.uniform(0, 2 * np.pi)
            t = np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5)
            g *= 1 + t * rng.uniform(0, 0.6)
        if rng.random() < 0.4:
            cx, cy = rng.uniform(-0.3, 1.3) * w, rng.uniform(-0.3, 1.3) * h
            d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (S * rng.uniform(0.5, 2)) ** 2
            g *= np.exp(-d2) * rng.uniform(0.2, 0.6) + 1
        L *= np.clip(g, 0.2, 3)[..., None]
    if rng.random() < 0.35:   # coloured light gradient (window + lamp)
        for c in range(3):
            L[..., c] *= 1 + smooth_field(rng, h, w, 2, rng.uniform(0, 0.05))
    if rng.random() < 0.3:   # curl / gutter / fold bands, along x or y
        horiz = rng.random() < 0.75
        n = w if horiz else h
        p = np.ones(n, np.float32); u = np.arange(n, dtype=np.float32)
        for _ in range(rng.randint(1, 3)):
            c = rng.choice([0, n - 1, rng.uniform(0, n)]); wd = n * rng.uniform(0.01, 0.2)
            p *= 1 - rng.uniform(0.05, 0.45) * np.exp(-0.5 * ((u - c) / wd) ** 2)
        L *= (p[None, :] if horiz else p[:, None])[..., None]
    nshadow = rng.choices([0, 1, 2, 3], [0.2, 0.5, 0.22, 0.08])[0]
    kinds = []
    for _ in range(nshadow):
        m, kind = occluder(rng, h, w)
        if m.mean() < 0.003 or m.mean() > 0.92:
            continue
        kinds.append(kind)
        m = penumbra(rng, m, S)
        a = rng.uniform(0.1, 0.65) if rng.random() < 0.85 else rng.uniform(0.6, 0.75)
        tint = np.float32([rng.gauss(0, 0.05) for _ in range(3)]); tint[0] -= abs(rng.gauss(0, 0.05))   # shadows lean blue
        a3 = np.clip(a * (1 + tint), 0.05, 0.78)
        var = 1 + smooth_field(rng, h, w, 3, rng.uniform(0, 0.1))
        L *= 1 - np.clip(m[..., None] * a3[None, None] * var[..., None], 0, 0.8)
    return L, kinds


# ---------------------------------------------------------------- samples
_cache = {}


def page_albedo(rng, h, w):
    p = rng.choice(PAGES)
    im = _cache.get(p)
    if im is None:
        im = cv2.imread(p)
        if len(_cache) > 40:
            _cache.pop(next(iter(_cache)))
        _cache[p] = im
    ih, iw = im.shape[:2]
    if (ih > iw) != (h > w) and rng.random() < 0.8:
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE); ih, iw = iw, ih
    s = max(h / ih, w / iw) * rng.uniform(1, 1.5)
    big = cv2.resize(im, (max(w, int(iw * s + 0.999)), max(h, int(ih * s + 0.999))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0 = rng.randint(0, big.shape[0] - h); x0 = rng.randint(0, big.shape[1] - w)
    a = big[y0:y0 + h, x0:x0 + w].astype(np.float32) / 255
    if rng.random() < 0.7:
        colourise(rng, a)
    return a


def sample(rng, h, w):
    a = page_albedo(rng, h, w)
    L, kinds = shading(rng, h // 2, w // 2)   # at half size (4x cheaper); edges stay ~1 px at full size
    L = cv2.resize(L, (w, h), interpolation=cv2.INTER_LINEAR)
    ref = np.percentile(cv2.resize(L, (w // 8, h // 8), interpolation=cv2.INTER_AREA).reshape(-1, 3), 99.5, axis=0)
    T = np.clip(L / ref, 0.02, 1.0)
    cast = np.float32([rng.uniform(0.9, 1.04) for _ in range(3)])
    expo = rng.uniform(0.75, 1.15)
    o = a * T
    o *= cast * np.float32(expo)
    if rng.random() < 0.6:   # camera / resampling blur
        o = cv2.GaussianBlur(o, (0, 0), rng.uniform(0.3, 1.1))
    np.clip(o, 0, 1, out=o)
    if rng.random() < 0.8:   # noise, stronger in the dark (shot noise) and chroma noise
        sig = np.float32(rng.uniform(0.002, 0.02) * 1.5)
        nz = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(o.shape, dtype=np.float32)
        o += nz * sig * np.sqrt(np.maximum(o, 0.02))
    np.clip(o, 0, 1, out=o)
    if rng.random() < 0.85:
        q = rng.randint(45, 95)
        _, enc = cv2.imencode('.jpg', (o * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        o = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255
    return o.astype(np.float32), T.astype(np.float32), a, kinds


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    rows = []
    for i in range(n):
        h, w = (512, 384) if rng.random() < 0.8 else (384, 512)
        o, T, a, k = sample(rng, h, w)
        tile = np.hstack([o, T, a])
        tile = (cv2.resize(tile, (int(tile.shape[1] * 300 / tile.shape[0]), 300)) * 255).astype(np.uint8)
        cv2.putText(tile, ' '.join(k), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tile = np.pad(tile, ((3, 3), (3, 3), (0, 0)), constant_values=80)
        rows.append(tile)
    W = max(t.shape[1] for t in rows)
    rows = [np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0)), constant_values=80) for t in rows]
    half = (len(rows) + 1) // 2
    L, R = np.vstack(rows[:half]), np.vstack(rows[half:] + ([np.full_like(rows[0], 80)] if len(rows) % 2 else []))
    cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else 'synth.jpg', np.hstack([L, R]))
