"""The shadow net's synthetic shading (tools/shadow_net/synth.py) with darker cast shadows.

The shipped net was trained on shadows that keep >= ~0.25 of the light (85 % of them 0.35-0.9), and the
app clamps the map at 0.3: a hand or leaf shadow at 0.23 of the paper (docres__for_deshadowing, the
Hi-TEC refill sheet) was only partly removed. Here a cast shadow keeps 0.12-0.9 of the light:
  55 %  0.4-0.9   (soft and medium shadows, as before)
  30 %  0.2-0.4   (hard shadows of a hand, a phone, leaves in sun)
  15 %  0.12-0.2  (the darkest a cast shadow on paper gets: direct sun, close occluder)
Dense foliage is drawn more often (it was the failing case). Everything else - page content, colour
blocks the net must keep, light falloff, curl bands, camera - is synth.py's.

  configure(root)        point synth.py at the data bundle (pages, BSDS photos)
  shading(rng, h, w)     -> (L float32 HxWx3, kinds)"""
import glob, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shadow_net'))
import synth  # noqa: E402

MIN_LIGHT = 0.12   # the darkest a cast shadow leaves


def configure(root, pages_dir):
    synth.BSDS = sorted(glob.glob(os.path.join(root, 'round4', 'assets', 'bsds', 'BSDS500', 'data', 'images', '*', '*.jpg')))
    synth.PAGES = sorted(glob.glob(os.path.join(pages_dir, '*.jpg')))
    if not synth.PAGES:
        raise SystemExit(f'no pages in {pages_dir} (make_bank.py pages)')
    if not synth.BSDS:
        raise SystemExit(f'no BSDS photos under {root}/round4/assets/bsds')


def darkness(rng):
    """the share of the light a cast shadow takes away"""
    r = rng.random()
    if r < 0.55:
        return rng.uniform(0.1, 0.6)
    if r < 0.85:
        return rng.uniform(0.6, 0.8)
    return rng.uniform(0.8, 1 - MIN_LIGHT)


def occluder(rng, h, w):
    m, kind = synth.occluder(rng, h, w)
    if kind != 'foliage' and rng.random() < 0.12:   # leaves more often: the failing case
        m2, k2 = synth.occluder(rng, h, w)
        if k2 == 'foliage':
            m, kind = np.maximum(m, m2), 'foliage'
    return m, kind


def shading(rng, h, w):
    """synth.shading with darker cast shadows (darkness()) and more foliage"""
    S = min(h, w)
    yy, xx = synth.grids(h, w)
    L = np.ones((h, w, 3), np.float32)
    if rng.random() < 0.8:   # smooth falloff: gradient, lamp, vignette
        g = 1 + synth.smooth_field(rng, h, w, rng.choice([2, 3, 4]), rng.uniform(0, 0.15))
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
            L[..., c] *= 1 + synth.smooth_field(rng, h, w, 2, rng.uniform(0, 0.05))
    if rng.random() < 0.3:   # curl / gutter / fold bands, along x or y
        horiz = rng.random() < 0.75
        n = w if horiz else h
        p = np.ones(n, np.float32); u = np.arange(n, dtype=np.float32)
        for _ in range(rng.randint(1, 3)):
            c = rng.choice([0, n - 1, rng.uniform(0, n)]); wd = n * rng.uniform(0.01, 0.2)
            p *= 1 - rng.uniform(0.05, 0.45) * np.exp(-0.5 * ((u - c) / wd) ** 2)
        L *= (p[None, :] if horiz else p[:, None])[..., None]
    nshadow = rng.choices([0, 1, 2, 3], [0.15, 0.5, 0.25, 0.1])[0]
    kinds = []
    for _ in range(nshadow):
        m, kind = occluder(rng, h, w)
        if m.mean() < 0.003 or m.mean() > 0.92:
            continue
        kinds.append(kind)
        m = synth.penumbra(rng, m, S)
        a = darkness(rng)
        tint = np.float32([rng.gauss(0, 0.05) for _ in range(3)]); tint[0] -= abs(rng.gauss(0, 0.05))   # shadows lean blue
        a3 = np.clip(a * (1 + tint), 0.05, 1 - MIN_LIGHT)
        var = 1 + synth.smooth_field(rng, h, w, 3, rng.uniform(0, 0.08))
        L *= 1 - np.clip(m[..., None] * a3[None, None] * var[..., None], 0, 1 - MIN_LIGHT)
    return L, kinds
