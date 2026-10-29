"""occluders.py: things in front of the page in a photo (gen.py): a thumb or fingers holding it, a hand lying on it, a pen,
a phone, a binder clip, a mug or a coin. The page finder's target stays the whole sheet (amodal: where a thumb covers
the page's edge, the edge goes on behind it, as the app wants its outline); SD-HR's target is the sheet's map as ever.

Each one is drawn analytically in centimetres round an anchor on a side of the page (local frame: U into the page, V
along the side), the page's projection taken as affine there (gen.py: its Jacobian at the anchor). Its shadow is the
same shape moved along the light by its height above the paper, with a soft edge growing with the height."""
import math

import numpy as np
import torch

# skin tones, light to deep (linear interpolation between neighbours)
SKIN = np.array([[0.95, 0.80, 0.70], [0.86, 0.66, 0.53], [0.74, 0.53, 0.40], [0.58, 0.40, 0.29], [0.40, 0.27, 0.19],
                 [0.27, 0.18, 0.13]])
KINDS = ('thumb', 'fingers', 'hand', 'pen', 'phone', 'clip', 'round')
KIND_P = (0.38, 0.14, 0.10, 0.15, 0.07, 0.09, 0.07)


def _skin(rng):
    t = rng.uniform(0, len(SKIN) - 1)
    k = min(int(t), len(SKIN) - 2)
    return np.clip(SKIN[k] * (1 - (t - k)) + SKIN[k + 1] * (t - k) + rng.normal(0, 0.02, 3), 0, 1)


def _cap(a, b, r0, r1, col, style, nail=None):
    return dict(t='cap', a=np.asarray(a, float), b=np.asarray(b, float), r0=r0, r1=r1, col=np.asarray(col, float), style=style, nail=nail)


def sample(rng, wcm):
    """one occluder (a dict: anchor side / u / turn, its parts in cm, height, blur); wcm: the page's width in cm"""
    kind = str(rng.choice(KINDS, p=KIND_P))
    r = rng.random()
    o = dict(kind=kind, wcm=wcm, side=int(rng.integers(4)), ang=float(np.clip(rng.normal(0, 0.3), -0.8, 0.8)),
             u=float(rng.uniform(0.02, 0.15) if r < 0.2 else rng.uniform(0.85, 0.98) if r < 0.4 else rng.uniform(0.1, 0.9)),
             blur=float(rng.uniform(0.5, 2.5)) if rng.random() < 0.4 else 0.0, dark=float(rng.uniform(0.25, 0.55)))
    prims = []
    if kind in ('thumb', 'fingers', 'hand'):
        sk = _skin(rng)
        nail = (rng.uniform(0.1, 0.9, 3) if rng.random() < 0.12 else None, rng.random() < 0.85)   # polish, nails seen
        sz = rng.uniform(0.85, 1.2)
        o['h'] = float(rng.uniform(0.6, 2.5))
        if kind == 'thumb':
            rr, d, L = 1.0 * sz, rng.uniform(0.2, 3.5), rng.uniform(5, 9)
            base = np.array([d - L, rng.normal(0, 0.5)])
            prims.append(_cap((d, 0), base, rr, rr * rng.uniform(1.1, 1.35), sk, 'skin', nail))
            if rng.random() < 0.75:   # the hand holding the page from behind, and its forearm: seen only past the paper
                th = math.atan2(base[1], base[0] - d) + rng.normal(0, 0.4)
                e = np.array([math.cos(th), math.sin(th)])
                pc = base + 4.0 * sz * e + np.array([-e[1], e[0]]) * rng.normal(0, 1.5)
                prims.append(dict(t='ell', c=pc, ax=(6.5 * sz, 4.6 * sz), rot=th, col=sk * 0.95, style='skin', under=True))
                prims.append(_cap(pc + 4.5 * sz * e, pc + 30 * e, 3.0 * sz, 3.8 * sz, sk * 0.93, 'skin'))
                prims[-1]['under'] = True
        elif kind == 'fingers':
            n = int(rng.integers(2, 5))
            rr, d = 0.8 * sz, rng.uniform(0.2, 2.5)
            stag = [0.3, 0.8, 0.5, -0.4][:n] if rng.random() < 0.5 else [-0.4, 0.5, 0.8, 0.3][4 - n:]
            for k in range(n):
                v = (k - (n - 1) / 2) * 2.1 * rr
                tip = d + stag[k] + rng.normal(0, 0.25)
                prims.append(_cap((tip, v), (tip - 8, v * 1.1), rr * rng.uniform(0.92, 1.05), rr * 1.15, sk, 'skin', nail))
            if rng.random() < 0.75:
                prims.append(dict(t='ell', c=np.array([d - 11.0, 0.0]), ax=(5.5 * sz, 4.8 * sz), rot=0.0, col=sk * 0.95, style='skin', under=True))
                prims.append(_cap((d - 14.0, 0.0), (d - 40.0, rng.normal(0, 4)), 3.0 * sz, 3.8 * sz, sk * 0.93, 'skin'))
                prims[-1]['under'] = True
        else:   # a hand lying flat: palm (outside the page or on it), four fingers into the page, the thumb aside
            pc = -rng.uniform(-4, 5)
            a, b = 4.6 * sz, 4.3 * sz
            prims.append(dict(t='ell', c=np.array([pc, 0.0]), ax=(a, b), rot=0.0, col=sk, style='skin'))
            prims.append(_cap((pc - 0.6 * a, 0.0), (pc - 32.0, rng.normal(0, 5)), 3.0 * sz, 3.8 * sz, sk * 0.95, 'skin'))   # the forearm
            fr = 0.85 * sz
            lens = [7.0, 7.8, 7.3, 6.0]
            for k in range(4):
                th = (k - 1.5) * 0.11 + rng.normal(0, 0.04)
                v0 = (k - 1.5) * 2.05 * fr
                base = np.array([pc + 0.45 * a, v0])
                tip = base + np.array([math.cos(th), math.sin(th)]) * lens[k] * sz
                prims.append(_cap(tip, base, fr, fr * 1.1, sk, 'skin', nail))
            sg = 1 if rng.random() < 0.5 else -1
            th = sg * rng.uniform(0.7, 1.1)
            base = np.array([pc - 0.1 * a, sg * 0.8 * b])
            prims.append(_cap(base + np.array([math.cos(th), math.sin(th)]) * 5.5 * sz, base, 1.0 * sz, 1.2 * sz, sk, 'skin', nail))
    elif kind == 'pen':
        o['h'] = float(rng.uniform(0.3, 0.7))
        L, rr = rng.uniform(12.5, 15), rng.uniform(0.35, 0.55)
        c = np.array([rng.uniform(-5, 5), 0.0])
        th = rng.uniform(-1.4, 1.4)
        e = np.array([math.cos(th), math.sin(th)])
        col = [np.full(3, rng.uniform(0.03, 0.12)), np.array([0.1, 0.15, 0.45]), np.array([0.6, 0.1, 0.1]),
               np.full(3, rng.uniform(0.75, 0.92)), rng.uniform(0.1, 0.9, 3)][int(rng.integers(5))]
        cap_col = col if rng.random() < 0.5 else rng.uniform(0.05, 0.9, 3)
        a, b = c - e * L / 2, c + e * L / 2
        prims.append(_cap(a + e * 4.5, b - e * 1.8, rr, rr, col, 'gloss'))
        prims.append(_cap(a, a + e * 4.5, rr * 1.08, rr * 1.08, cap_col, 'gloss'))
        prims.append(_cap(b - e * 1.8, b, rr, rr * 0.12, np.full(3, 0.7) if rng.random() < 0.5 else col, 'gloss'))
    elif kind == 'phone':
        o['h'] = float(rng.uniform(0.7, 1.0))
        s = rng.uniform(0.9, 1.1)
        prims.append(dict(t='rrect', c=np.array([rng.uniform(-6, 2), rng.uniform(-3, 3)]), half=(7.5 * s, 3.6 * s), rad=0.9 * s,
                          rot=rng.uniform(-math.pi, math.pi), col=np.full(3, rng.uniform(0.02, 0.1)), style='phone',
                          case=rng.uniform(0.05, 0.85, 3) if rng.random() < 0.6 else np.full(3, rng.uniform(0.02, 0.1)),
                          glow=rng.uniform(0.0, 0.18), streak=rng.uniform(-3, 3)))
    elif kind == 'clip':
        o['h'] = float(rng.uniform(0.4, 0.8))
        s = rng.uniform(0.6, 1.3)
        dep = rng.uniform(0.8, 1.6) * s
        prims.append(dict(t='rrect', c=np.array([(dep - 0.6 * s) / 2, 0.0]), half=((dep + 0.6 * s) / 2, 1.6 * s), rad=0.15 * s, rot=0.0,
                          col=np.full(3, rng.uniform(0.02, 0.1)) if rng.random() < 0.7 else rng.uniform(0.1, 0.8, 3), style='matte'))
        for sg in (-1, 1):
            prims.append(_cap((-0.6 * s, sg * 1.1 * s), (-0.6 * s - rng.uniform(1.5, 2.2) * s, sg * rng.uniform(0.7, 1.1) * s),
                              0.08 * s, 0.08 * s, np.full(3, 0.75), 'gloss'))
    else:   # round: a mug from above, or a coin
        if rng.random() < 0.6:
            rr = rng.uniform(3.8, 4.6)
            o['h'] = float(rng.uniform(3, 9))
            prims.append(dict(t='disc', c=np.array([rng.uniform(-rr, rr), 0.0]), r=rr, style='mug',
                              col=rng.uniform(0.1, 0.95, 3) if rng.random() < 0.5 else np.full(3, rng.uniform(0.8, 0.95)),
                              fill=np.array([0.16, 0.09, 0.05]) * rng.uniform(0.6, 1.4) if rng.random() < 0.7 else None))
        else:
            rr = rng.uniform(0.9, 1.4)
            o['h'] = 0.15
            prims.append(dict(t='disc', c=np.array([rng.uniform(-rr, rr), rng.uniform(-3, 3)]), r=rr, style='coin',
                              col=np.array([0.72, 0.6, 0.35]) if rng.random() < 0.4 else np.full(3, 0.68)))
    o['prims'] = prims
    return o


def extent(o):
    """local-frame points (cm) covering the occluder"""
    pts = []
    for p in o['prims']:
        if p['t'] == 'cap':
            r = max(p['r0'], p['r1'])
            pts += [p['a'] + [sx * r, sy * r] for sx in (-1, 1) for sy in (-1, 1)] + [p['b'] + [sx * r, sy * r] for sx in (-1, 1) for sy in (-1, 1)]
        else:
            r = max(p['ax']) if p['t'] == 'ell' else (p['r'] if p['t'] == 'disc' else math.hypot(*p['half']))
            pts += [p['c'] + [sx * r, sy * r] for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(pts)


def _prim(p, U, V):
    """signed distance (cm, < 0 inside), and shading terms: nz (height of a round body), lat (-1..1 across), al (cm along
    from the tip / 'a')"""
    if p['t'] == 'cap':
        a, b = p['a'], p['b']
        ab = b - a
        L2 = float(ab @ ab) + 1e-9
        h = (((U - a[0]) * ab[0] + (V - a[1]) * ab[1]) / L2).clamp(0, 1)
        du, dv = U - (a[0] + h * ab[0]), V - (a[1] + h * ab[1])
        r = p['r0'] + (p['r1'] - p['r0']) * h
        dc = torch.sqrt(du * du + dv * dv)
        L = math.sqrt(L2)
        lat = ((du * -ab[1] + dv * ab[0]) / L / r.clamp_min(1e-6)).clamp(-1, 1)
        return dc - r, torch.sqrt((1 - (dc / r) ** 2).clamp_min(0)), lat, h * L
    if p['t'] == 'ell':
        c, s = math.cos(p['rot']), math.sin(p['rot'])
        x, y = U - p['c'][0], V - p['c'][1]
        x, y = x * c + y * s, -x * s + y * c
        rho = torch.sqrt((x / p['ax'][0]) ** 2 + (y / p['ax'][1]) ** 2)
        return (rho - 1) * min(p['ax']), torch.sqrt((1 - rho ** 2).clamp_min(0)) ** 0.5, (y / p['ax'][1]).clamp(-1, 1), x
    if p['t'] == 'rrect':
        c, s = math.cos(p['rot']), math.sin(p['rot'])
        x, y = U - p['c'][0], V - p['c'][1]
        x, y = x * c + y * s, -x * s + y * c
        qx = x.abs() - (p['half'][0] - p['rad'])
        qy = y.abs() - (p['half'][1] - p['rad'])
        d = torch.sqrt(qx.clamp_min(0) ** 2 + qy.clamp_min(0) ** 2) + torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx)) - p['rad']
        return d, torch.ones_like(d), (y / p['half'][1]).clamp(-1, 1), x
    x, y = U - p['c'][0], V - p['c'][1]
    dc = torch.sqrt(x * x + y * y)
    return dc - p['r'], torch.sqrt((1 - (dc / p['r']) ** 2).clamp_min(0)), (y / p['r']).clamp(-1, 1), x


def _shade(p, d, nz, lat, al, lv):
    """colour 3 x h x w of part p where it is the nearest part"""
    col = torch.tensor(p['col'], device=d.device, dtype=torch.float32)[:, None, None]
    st = p['style']
    if st == 'skin':
        s = 0.55 + 0.45 * nz + 0.12 * lat * lv
        rim = (1 - nz)
        c = col * s * (1 - 0.12 * rim) + torch.tensor([0.05, -0.01, -0.02], device=d.device)[:, None, None] * rim
        if p['t'] == 'cap':
            r = p['r0']
            for j in (2.4 * r, 4.4 * r):                               # joint creases
                c = c * (1 - 0.1 * torch.exp(-((al - j) / (0.06 * r + 0.02)) ** 2))[None]
            if p.get('nail') is not None and p['nail'][1]:
                pol = p['nail'][0]
                en = ((al - 0.62 * r) / (0.62 * r)) ** 2 + (lat / 0.62) ** 2
                inn = (1 - en).clamp(0, 1).sqrt()
                ncol = (col * 0.55 + 0.42) if pol is None else torch.tensor(pol, device=d.device, dtype=torch.float32)[:, None, None]
                nm = (en < 1).float() * (al > 0.05 * r).float()
                c = c * (1 - nm) + nm * ncol * (0.8 + 0.2 * inn + 0.25 * inn ** 8)
        return c + 0.05 * nz ** 10
    if st == 'gloss':
        hl = torch.exp(-((lat - 0.35 * (1 if lv >= 0 else -1)) / 0.13) ** 2)
        return col * (0.45 + 0.55 * nz) + 0.45 * hl
    if st == 'matte':
        return col * (0.75 + 0.25 * nz) + 0.06 * (lat * lv)
    if st == 'phone':
        case = torch.tensor(p['case'], device=d.device, dtype=torch.float32)[:, None, None]
        rim = (d > -0.3).float()
        glass = col + p['glow'] + 0.1 * torch.exp(-((al * 0.3 + lat * 2 - p['streak']) / 0.5) ** 2)    # a reflection streak
        return case * rim + glass * (1 - rim)
    if st == 'mug':
        rim = (d > -0.45).float()
        r = p['r']
        inner = (col * 0.75 if p['fill'] is None else torch.tensor(p['fill'], device=d.device, dtype=torch.float32)[:, None, None]) \
            * (0.75 + 0.25 * (lat * lv + 1) / 2)
        return col * (0.8 + 0.2 * nz) * rim + inner * (1 - rim)
    # coin
    return col * (0.6 + 0.5 * nz ** 0.5) + 0.04 * torch.sin(al * 9 + lat * 7)


def draw(o, U, V, Us, Vs, px_cm, lu, lv):
    """o: sample(); U, V: h x w local cm of the pixels; Us, Vs: the same moved back along the light (its shadow);
    px_cm: a pixel's size in cm; (lu, lv): the light's direction along U / V. -> colour 3 x h x w (unlit: gen.py
    lights it), coverage h x w, shadow h x w, under h x w (1: a part behind the page, seen only past the paper)"""
    best = None
    for p in o['prims']:
        d, nz, lat, al = _prim(p, U, V)
        c = _shade(p, d, nz, lat, al, lv)
        u = torch.full_like(d, float(p.get('under', False)))
        if best is None:
            best, col, und = d, c, u
        else:
            take = d < best
            col = torch.where(take[None], c, col)
            und = torch.where(take, u, und)
            if p['style'] == 'skin':   # skin parts join smoothly (a thumb into its hand)
                hk = (0.5 + 0.5 * (d - best) / 0.8).clamp(0, 1)
                best = d * (1 - hk) + best * hk - 0.8 * hk * (1 - hk)
            else:
                best = torch.minimum(best, d)
    cov = (0.5 - best / px_cm).clamp(0, 1)
    ds = torch.stack([_prim(p, Us, Vs)[0] for p in o['prims'] if not p.get('under')]).min(0).values
    w = max(px_cm, 0.12 * o['h'] + 0.05)
    sh = (0.5 - ds / w).clamp(0, 1)
    # a little skin / surface mottling
    g = torch.randn(1, 1, 6, 6, device=U.device)
    mot = torch.nn.functional.interpolate(g, size=U.shape, mode='bicubic', align_corners=False)[0, 0]
    return col * (1 + 0.025 * mot)[None], cov, sh, und
