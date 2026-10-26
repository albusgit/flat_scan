"""geom.py: paper shapes, a camera and a z-buffered rasteriser, with exact ground truth for the flattening nets.

A sheet of paper W x H (W = 1, H = the paper's height / width) is a map S(s, t) -> R^3 of its page coordinates
(s, t) in [0, 1]^2. The shapes are isometric (paper bends, it does not stretch) except `wave` (a small non-isometric
ripple for nearly flat paper):
  flat      S = (s W, t H, 0)
  curl      the sheet bent round a cylinder beyond a line (a book page, a rolled receipt): radius R, from the line on
  corner    a corner lifted: the part beyond a line cutting two adjacent sides bent round a cylinder
  fold      a straight fold through the sheet at an angle (a letter folded and opened), slightly rounded
  wave      a gentle ripple (cockled paper)
Several can be chained (each bends the previous result about its own line in the sheet's own coordinates; the
first bend's line is in page coordinates, later ones in the bent sheet's - kept simple: bends are applied as
rigid rotations of the sheet's parts about lines lying in the z = 0 plane before any bend, in order).

camera(): a pinhole looking at the sheet from above with tilt, turn and spin; project(P) -> photo px, depth.
rasterise(V, F, uv, size): per photo pixel the visible face's page coordinates (s, t) (NaN: background) and its
depth and normal (for shading). Torch, CPU or GPU, fully vectorised over triangles x their bounding-box pixels.
The ground-truth backward map of the page (what the net predicts) is simply project(S(grid)) - exact, also where
the paper hides itself.
"""
import math

import numpy as np
import torch


# ---------------------------------------------------------------- shapes

def bend(P, s_dir, c, R, phi, keep_side=-1):
    """Bend the points P (N x 3, sheet still flat where this bend acts: z of the bent part is added) round a cylinder.
    s_dir: unit 2-vector in the sheet plane, the bend's normal; c: the line's offset (points with d = P.s_dir - c > 0
    bend); R: radius; phi: the largest angle (beyond the arc d > R phi the sheet goes on straight, tangent).
    Isometric for a flat sheet: arc length along s_dir is kept."""
    n = np.array([s_dir[0], s_dir[1], 0.0])
    d = P[:, :2] @ np.asarray(s_dir) - c
    out = P.copy()
    m = d > 0
    dd = d[m]
    arc = np.minimum(dd, R * phi)
    rest = np.maximum(dd - R * phi, 0)
    # in the bend's (along n, up z) plane: the arc, then the tangent line
    a = arc / R
    along = R * np.sin(a) + rest * np.cos(phi)
    up = R * (1 - np.cos(a)) + rest * np.sin(phi)
    out[m, 0] += (along - dd) * n[0]
    out[m, 1] += (along - dd) * n[1]
    out[m, 2] += up
    return out


def fold(P, s_dir, c, angle, round_r=0.004):
    """A fold: the part beyond the line turned up by |angle| (rad) about it, with a small rounding radius (paper
    lies on a table: a fold lifts one side; which side is the line's normal's choice)."""
    return bend(P, s_dir, c, max(round_r, 1e-4), abs(angle))


def wave(P, rng, amp):
    """A smooth ripple: z += amp x a sum of 2-3 random sinusoids (not isometric: for nearly flat paper only)."""
    z = np.zeros(len(P))
    for _ in range(int(rng.integers(2, 4))):
        k = rng.uniform(2, 7) * 2 * math.pi
        th = rng.uniform(0, math.pi)
        ph = rng.uniform(0, 2 * math.pi)
        z += np.sin(k * (P[:, 0] * math.cos(th) + P[:, 1] * math.sin(th)) + ph) * rng.uniform(0.3, 1)
    out = P.copy()
    out[:, 2] += amp * z / 3
    return out


def random_shape(rng, H, kind=None):
    """kind (or a random one) -> (name, f) with f(P) the deformation of flat sheet points P (N x 3). Every random
    choice is made here, once: f is a fixed shape (it is called for the render mesh and the ground-truth grids)."""
    kinds = ['flat', 'nearflat', 'curl', 'corner', 'fold', 'fold2', 'curl_corner', 'wave_curl']
    w = [0.16, 0.14, 0.18, 0.16, 0.12, 0.06, 0.10, 0.08]
    kind = kind or str(rng.choice(kinds, p=np.array(w) / sum(w)))
    L = max(1.0, H)
    steps = []   # (op, args) applied in order

    def line_through_corner():
        # a line cutting two adjacent sides near a corner: its normal points at the corner
        cx, cy = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        corner = np.array([cx * 1.0, cy * H])
        a = rng.uniform(0.08, 0.45) * min(1.0, H) * 1.6
        b = rng.uniform(0.08, 0.45) * min(1.0, H) * 1.6
        p1 = corner + np.array([(1 - 2 * cx) * a, 0])
        p2 = corner + np.array([0, (1 - 2 * cy) * b])
        t = p2 - p1
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        if (corner - p1) @ n < 0:
            n = -n
        return n, float(p1 @ n)

    def random_line(edge_frac=(0.15, 0.85)):
        th = rng.uniform(0, 2 * math.pi)
        n = np.array([math.cos(th), math.sin(th)])
        p = np.array([rng.uniform(*edge_frac), rng.uniform(*edge_frac) * H])
        return n, float(p @ n)

    def axis_line(lo, hi):
        ax = int(rng.integers(0, 2))
        n = np.array([1.0, 0.0]) if ax == 0 else np.array([0.0, 1.0])
        ext = 1.0 if ax == 0 else H
        c0 = rng.uniform(lo, hi) * ext
        if rng.random() < 0.5:   # bend the other end
            return -n, -(ext - c0)
        return n, c0

    def wave_params(amp):
        return [(rng.uniform(2, 7) * 2 * math.pi, rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 1))
                for _ in range(int(rng.integers(2, 4)))], amp

    if kind == 'nearflat':
        steps.append(('wave', wave_params(rng.uniform(0.002, 0.012) * L)))
    elif kind in ('curl', 'wave_curl'):
        n, c = axis_line(0.1, 0.7)
        steps.append(('bend', (n, c, rng.uniform(0.15, 1.2) * L, rng.uniform(0.3, 1.6))))
        if kind == 'wave_curl':
            steps.append(('wave', wave_params(rng.uniform(0.002, 0.008) * L)))
    elif kind == 'corner':
        n, c = line_through_corner()
        steps.append(('bend', (n, c, rng.uniform(0.03, 0.4) * min(1.0, H), rng.uniform(0.3, 2.2))))
    elif kind == 'curl_corner':
        n, c = axis_line(0.3, 0.8)
        steps.append(('bend', (n, c, rng.uniform(0.4, 1.5) * L, rng.uniform(0.2, 0.8))))
        n2, c2 = line_through_corner()
        steps.append(('bend', (n2, c2, rng.uniform(0.05, 0.3) * min(1.0, H), rng.uniform(0.3, 1.5))))
    elif kind == 'fold':
        n, c = random_line()
        steps.append(('fold', (n, c, rng.uniform(0.1, 0.9))))
    elif kind == 'fold2':
        ax = int(rng.integers(0, 2))
        n = np.array([1.0, 0.0]) if ax == 0 else np.array([0.0, 1.0])
        ext = 1.0 if ax == 0 else H
        steps.append(('fold', (n, ext / 3, rng.uniform(0.1, 0.6))))
        steps.append(('fold', (n, 2 * ext / 3, rng.uniform(0.1, 0.6))))
    elif kind != 'flat':
        raise ValueError(kind)

    def f(P):
        for op, args in steps:
            if op == 'bend':
                P = bend(P, *args)
            elif op == 'fold':
                P = fold(P, *args)
            else:
                comps, amp = args
                z = np.zeros(len(P))
                for k, th, ph, a in comps:
                    z += np.sin(k * (P[:, 0] * math.cos(th) + P[:, 1] * math.sin(th)) + ph) * a
                P = P.copy()
                P[:, 2] += amp * z / 3
        return P
    return kind, f


def sheet(H, nu, nv):
    """the flat sheet's mesh: vertices (nv x nu x 3), page coords (nv x nu x 2), faces (2 per quad)"""
    s = np.linspace(0, 1, nu)
    t = np.linspace(0, 1, nv)
    S, T = np.meshgrid(s, t)
    P = np.stack([S, T * H, np.zeros_like(S)], -1)
    idx = np.arange(nu * nv).reshape(nv, nu)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return P, np.stack([S, T], -1), F


# ---------------------------------------------------------------- camera

class Camera:
    """pinhole; the sheet's frame (x right, y down the page, z up off the table) is turned into a right-handed frame
    looking into the table (z down) before the rotation R and translation t"""
    def __init__(self, Rm, tvec, f, cx, cy, size):
        self.R, self.t, self.f, self.cx, self.cy, self.size = Rm, tvec, f, cx, cy, size

    def project(self, P):
        """P: N x 3 sheet frame -> (N x 2 photo px, N depth)"""
        X = (P * np.array([1.0, 1.0, -1.0])) @ self.R.T + self.t
        z = X[:, 2]
        return np.stack([self.f * X[:, 0] / z + self.cx, self.f * X[:, 1] / z + self.cy], 1), z


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_camera(rng, P, size, fill=(0.6, 1.05), tilt=35, spin=12, upright=True):
    """A camera above the deformed sheet P (N x 3): tilted up to `tilt` deg about x, 0.6 x that about y, turned in
    plane by up to `spin` deg (and by k x 90 deg when not upright), the sheet's extent filling `fill` of the photo's
    shorter side, placed off-centre."""
    W, Hh = size
    f = rng.uniform(0.65, 1.1) * max(W, Hh)
    a_t = math.radians(rng.uniform(-tilt, tilt))
    a_u = math.radians(rng.uniform(-tilt, tilt) * 0.6)
    a_s = math.radians(rng.uniform(-spin, spin)) + (0 if upright else math.pi / 2 * int(rng.integers(0, 4)))
    Rm = rot('z', a_s) @ rot('x', a_t) @ rot('y', a_u)
    ctr = P.mean(0) * np.array([1.0, 1.0, -1.0])
    ext = np.linalg.norm(P[:, :2].max(0) - P[:, :2].min(0))
    dist = f * ext / (rng.uniform(*fill) * min(W, Hh))
    tvec = -Rm @ ctr + np.array([0, 0, dist])
    cx = W / 2 + rng.uniform(-0.1, 0.1) * W
    cy = Hh / 2 + rng.uniform(-0.1, 0.1) * Hh
    return Camera(Rm, tvec, f, cx, cy, size)


# ---------------------------------------------------------------- rasteriser

def rasterise(V2, z, F, uv, size, device='cpu', max_box=24):
    """V2: N x 2 photo px, z: N depths, F: M x 3 faces, uv: N x 2 page coords, size (W, H) ->
    (st H x W x 2 float32, NaN = background; face H x W int64, -1 = background).
    Triangles larger than max_box px across are split by the caller (use a finer mesh)."""
    W, H = size
    V2 = torch.as_tensor(V2, dtype=torch.float32, device=device)
    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    F_ = torch.as_tensor(F, dtype=torch.long, device=device)
    uv = torch.as_tensor(uv, dtype=torch.float32, device=device)
    A, B, C = V2[F_[:, 0]], V2[F_[:, 1]], V2[F_[:, 2]]
    lo = torch.floor(torch.minimum(torch.minimum(A, B), C)).long()
    hi = torch.ceil(torch.maximum(torch.maximum(A, B), C)).long()
    span = (hi - lo).clamp(min=0)
    keep = (span[:, 0] < max_box) & (span[:, 1] < max_box) & (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < W) & (lo[:, 1] < H)
    fi = torch.nonzero(keep).squeeze(1)
    zbuf = torch.full((H * W,), float('inf'), device=device)
    best = torch.full((H * W,), -1, dtype=torch.long, device=device)
    st = torch.full((H * W, 2), float('nan'), device=device)
    # pixels per face: the box's grid, in chunks of faces
    k = max_box
    oy, ox = torch.meshgrid(torch.arange(k, device=device), torch.arange(k, device=device), indexing='ij')
    ox, oy = ox.reshape(-1), oy.reshape(-1)
    records = []
    for chunk in torch.split(fi, max(1, 4_000_000 // (k * k))):
        px = lo[chunk, 0:1] + ox[None]           # faces x k*k
        py = lo[chunk, 1:2] + oy[None]
        inb = (px <= hi[chunk, 0:1]) & (py <= hi[chunk, 1:2]) & (px >= 0) & (py >= 0) & (px < W) & (py < H)
        qx, qy = px.float() + 0.5, py.float() + 0.5
        a, b, c = A[chunk], B[chunk], C[chunk]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
        w0 = ((b[:, 1:2] - c[:, 1:2]) * (qx - c[:, 0:1]) + (c[:, 0:1] - b[:, 0:1]) * (qy - c[:, 1:2])) / den[:, None]
        w1 = ((c[:, 1:2] - a[:, 1:2]) * (qx - c[:, 0:1]) + (a[:, 0:1] - c[:, 0:1]) * (qy - c[:, 1:2])) / den[:, None]
        w2 = 1 - w0 - w1
        e = -1e-4
        inside = inb & (w0 >= e) & (w1 >= e) & (w2 >= e)
        f3 = F_[chunk]
        zz = w0 * z[f3[:, 0:1]] + w1 * z[f3[:, 1:2]] + w2 * z[f3[:, 2:3]]
        pix = (py * W + px)[inside]
        zv = zz[inside]
        zbuf.scatter_reduce_(0, pix, zv, reduce='amin')
        records.append((pix, zv, chunk[:, None].expand_as(px)[inside], w0[inside], w1[inside], w2[inside]))
    for pix, zv, face, w0, w1, w2 in records:
        win = zv <= zbuf[pix] + 1e-7
        p, f3 = pix[win], F_[face[win]]
        st[p] = w0[win, None] * uv[f3[:, 0]] + w1[win, None] * uv[f3[:, 1]] + w2[win, None] * uv[f3[:, 2]]
        best[p] = face[win]
    return st.reshape(H, W, 2).cpu().numpy(), best.reshape(H, W).cpu().numpy()


def face_normals(P3, F):
    a, b, c = P3[F[:, 0]], P3[F[:, 1]], P3[F[:, 2]]
    n = np.cross(b - a, c - a)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
