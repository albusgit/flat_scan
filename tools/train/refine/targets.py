"""Refinement targets by composition (round 20 pilot).

Stage 1 (the nets, no line stage) gives a backward map B: stage-1 frame px -> photo px, on a 4-px grid (FS_DUMP_MAP).
The synthetic generator gives the exact map G: page point u = (s, t) in [0, 1]^2 -> photo px. A refiner predicts a
correction field d on the stage-1 frame so that the refined map B(x + d(x)) shows page point A^-1(x) at x, where A is
the best affine page -> frame (position, scale, rotation, aspect and shear are not the refiner's job):

    d(x) = B^-1(G(A^-1(x))) - x

on the centres of CELL-px cells. Nothing is subtracted between maps: the target is exact wherever B can be inverted.
"""
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

STEP = 4     # B's grid spacing in stage-1 px
CELL = 32    # target cell (the refiner's output stride)


class Backward:
    """B on its 4-px grid, read bilinearly, and its inverse (a linear guess on every 4th grid point, then Newton)"""
    def __init__(self, bx, by):
        self.bx, self.by = np.asarray(bx, np.float64), np.asarray(by, np.float64)
        self.gh, self.gw = self.bx.shape
        self.W, self.H = STEP * (self.gw - 1), STEP * (self.gh - 1)
        sx, sy = self.bx[::4, ::4], self.by[::4, ::4]
        ok = np.isfinite(sx) & np.isfinite(sy)
        oi, oj = np.nonzero(ok)
        self.guess = LinearNDInterpolator(np.stack([sx[ok], sy[ok]], 1), np.stack([16.0 * oj, 16.0 * oi], 1))

    def __call__(self, v):
        c = [v[:, 1] / STEP, v[:, 0] / STEP]
        return np.stack([map_coordinates(self.bx, c, order=1, mode='nearest'),
                         map_coordinates(self.by, c, order=1, mode='nearest')], 1)

    def inverse(self, p, iters=6, h=2.0):
        """p: N x 2 photo px -> (N x 2 frame px, N ok): ok where Newton converged to 0.1 px inside the frame"""
        v = self.guess(p)
        ok = np.isfinite(v).all(1)
        v[~ok] = 0
        for _ in range(iters):
            r = p - self(v)
            jx = (self(v + [h, 0]) - self(v - [h, 0])) / (2 * h)
            jy = (self(v + [0, h]) - self(v - [0, h])) / (2 * h)
            det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
            det = np.where(np.abs(det) < 1e-9, np.nan, det)
            v = v + np.stack([(jy[:, 1] * r[:, 0] - jy[:, 0] * r[:, 1]) / det,
                              (jx[:, 0] * r[:, 1] - jx[:, 1] * r[:, 0]) / det], 1)
            v = np.nan_to_num(v, nan=-1e6)
        res = np.linalg.norm(p - self(v), axis=1)
        inside = (v[:, 0] >= 0) & (v[:, 0] <= self.W) & (v[:, 1] >= 0) & (v[:, 1] <= self.H)
        return v, ok & inside & (res < 0.1)


def page_to_photo(G, u):
    """G: 2 x GH x GW (photo px of s = j / (GW - 1), t = i / (GH - 1)); u: N x 2 (s, t) -> N x 2 photo px"""
    GH, GW = G.shape[1:]
    c = [u[:, 1] * (GH - 1), u[:, 0] * (GW - 1)]
    return np.stack([map_coordinates(G[0].astype(np.float64), c, order=1, mode='nearest'),
                     map_coordinates(G[1].astype(np.float64), c, order=1, mode='nearest')], 1)


def build(bx, by, G, cell=CELL, n_fit=48):
    """-> dict(d: ny x nx x 2 float32 frame px, valid: ny x nx bool, M: 3 x 2 affine ([s, t, 1] @ M = frame px),
    frame: (W, H), stats) or None when stage 1 lost the page (under half the page points land in its frame)"""
    Bk = Backward(bx, by)
    GH, GW = G.shape[1:]
    m = max(8, int(round(n_fit * (GH - 1) / (GW - 1))))
    s, t = np.meshgrid(np.linspace(0, 1, n_fit), np.linspace(0, 1, m))
    u = np.stack([s.ravel(), t.ravel()], 1)
    v, ok = Bk.inverse(page_to_photo(G, u))
    if ok.mean() < 0.5:
        return None
    U = np.concatenate([u, np.ones((len(u), 1))], 1)
    M, *_ = np.linalg.lstsq(U[ok], v[ok], rcond=None)
    nx, ny = int(np.ceil((Bk.W + 1) / cell)), int(np.ceil((Bk.H + 1) / cell))
    X, Y = np.meshgrid(cell * (np.arange(nx) + 0.5), cell * (np.arange(ny) + 0.5))
    x = np.stack([X.ravel(), Y.ravel()], 1)
    uu = (x - M[2]) @ np.linalg.inv(M[:2])
    inpage = (uu >= 0).all(1) & (uu <= 1).all(1) & (x[:, 0] <= Bk.W) & (x[:, 1] <= Bk.H)
    v2, ok2 = Bk.inverse(page_to_photo(G, np.clip(uu, 0, 1)))
    valid = inpage & ok2
    d = np.where(valid[:, None], v2 - x, 0).reshape(ny, nx, 2).astype(np.float32)
    valid = valid.reshape(ny, nx)
    # stretch: |d's difference between neighbouring cells| / cell (what bends text rows)
    dx = np.linalg.norm(np.diff(d, axis=1), axis=-1)[valid[:, 1:] & valid[:, :-1]] / cell
    dy = np.linalg.norm(np.diff(d, axis=0), axis=-1)[valid[1:] & valid[:-1]] / cell
    st = np.concatenate([dx, dy])
    n = np.linalg.norm(d[valid], axis=-1)
    stats = dict(fit_share=float(ok.mean()), valid_share=float(valid.mean()), rms=float(np.sqrt(np.mean(n ** 2))),
                 p95=float(np.percentile(n, 95)), stretch_med=float(np.median(st)), stretch_p90=float(np.percentile(st, 90)))
    return dict(d=d, valid=valid, M=M.astype(np.float32), frame=(Bk.W, Bk.H), stats=stats)


def render(photo, bx, by):
    """the stage-1 page the refiner sees: the photo through B at every frame px ((W + 1) x (H + 1))"""
    gh, gw = bx.shape
    W, H = STEP * (gw - 1) + 1, STEP * (gh - 1) + 1
    gx = np.tile(np.arange(W, dtype=np.float32) / STEP, (H, 1))
    gy = np.tile((np.arange(H, dtype=np.float32) / STEP)[:, None], (1, W))
    bx32 = np.nan_to_num(np.asarray(bx, np.float32), nan=-1e4)
    by32 = np.nan_to_num(np.asarray(by, np.float32), nan=-1e4)
    mx = cv2.remap(bx32, gx, gy, cv2.INTER_LINEAR)
    my = cv2.remap(by32, gx, gy, cv2.INTER_LINEAR)
    return cv2.remap(photo, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
