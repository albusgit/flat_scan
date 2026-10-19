"""Audit of the reference solves: rebuild the exact sparse systems of warpmap.local_field from
the exported samples and compare the reference fields (pass 2: CG + smoothed-aggregation AMG,
tol 1e-5, warm start; pass 1 / global: SuperLU) with a tight direct solve of the same system.
Also writes the systems as fixtures (CSR + rhs) for the C++ solver tests.
usage: python -I tools/solver_audit.py REF_CODE_DIR FIXTURE_DIR TAG [--save]"""
import sys, os
code, fx, tag = sys.argv[1:4]
sys.path.insert(0, os.path.abspath(code))
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, warpmap as WM
z = np.load(f'{fx}/{tag}_warp.npz')
D = z['D']; h, w = D.shape
def system(G, sxy, vals, wts):
    WM.G = G
    gw, gh = w // G + 2, h // G + 2
    Lm = WM.regularizer(gw, gh)
    A = WM.bilinear_matrix(sxy[:, 0], sxy[:, 1], gw, gh)
    R = (Lm.T @ Lm + 1e-9 * sp.eye(gw * gh)).tocsr()
    M = (A.T @ sp.diags(wts) @ A + R).tocsr()
    return M, A.T @ (wts * vals), gw, gh
T = z['T']
out = {}
# pass 2 (fine grid, warm-started AMG in the reference)
sh = np.r_[z['S2_h'], T]; wts = np.ones(len(sh))
xs = sh[:, 0] - WM.grid_sample(z['Fx1'], WM.G_COARSE, sh[:, 0], sh[:, 1])
My, by, gw, gh = system(8, np.c_[xs, sh[:, 1]], sh[:, 2], wts)
sv = z['S2_v']
ys = sv[:, 0] - WM.grid_sample(z['Fy1'], WM.G_COARSE, sv[:, 1], sv[:, 0])
Mx, bx, _, _ = system(8, np.c_[sv[:, 1], ys], sv[:, 2], np.ones(len(sv)))
for name, M, b, Fref in (('pass2_y', My, by, z['Fy']), ('pass2_x', Mx, bx, z['Fx'])):
    ex = spla.spsolve(M.tocsc(), b).reshape(gh, gw)
    r_ref = np.linalg.norm(b - M @ Fref.ravel()) / np.linalg.norm(b)
    # field difference in px, on the grid nodes inside the image
    d = np.abs(ex - Fref)[: h // 8 + 1, : w // 8 + 1]
    print(f'{tag} {name}: n={M.shape[0]}  reference rel. residual {r_ref:.2e}  |AMG - exact| max {d.max():.4f} px, p99 {np.percentile(d, 99):.4f} px')
    out[name] = (M, b, Fref, ex)
if '--save' in sys.argv:
    for name, (M, b, Fref, ex) in out.items():
        M = M.tocsr(); M.sort_indices()
        np.savez(f'{fx}/{tag}_{name}_system.npz', indptr=M.indptr.astype(np.int64), indices=M.indices.astype(np.int64),
                 data=M.data, b=b, ref=Fref.ravel(), exact=ex.ravel())
