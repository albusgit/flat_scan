"""Stage-by-stage comparison of the C++ warp-map port with the reference fixtures.
usage: python -I tools/compare_warp.py FIXTURE_WARP.npz CANDIDATE.npz [REF_A_IMG.png CAND_A_IMG.png]"""
import sys, numpy as np
ref, cand = np.load(sys.argv[1]), np.load(sys.argv[2])
def rowdiff(name):
    a, b = ref[name], cand[name]
    if a.shape != b.shape:
        print(f'{name:6s} shape ref {a.shape} cand {b.shape}  <-- DIFFERENT'); return False
    d = np.abs(a - b).max() if a.size else 0.0
    print(f'{name:6s} n={len(a):6d} max |diff| {d:.3g}'); return True
if len(sys.argv) > 4:
    import cv2
    A, B = cv2.imread(sys.argv[3]).astype(int), cv2.imread(sys.argv[4]).astype(int)
    print(f'A_img  max |diff| {np.abs(A - B).max()} levels, pixels differing {(np.abs(A - B).max(2) > 0).sum()}')
print(f'ctrl   ref raw/decimated {len(ref["ctrl_dst_raw"])}/700?  cand {cand["n_ctrl"].astype(int).tolist()}')
for f in ('GX', 'GY'):
    print(f'{f:6s} max |diff| {np.abs(ref[f] - cand[f]).max():.3g} px')
print(f'D      max |diff| {np.abs(ref["D"] - cand["D"]).max():.3g}')
for n in ('H', 'V', 'S1_h', 'S1_v', 'T'):
    rowdiff(n)
for f in ('Fx1', 'Fy1'):
    print(f'{f:6s} max |diff| {np.abs(ref[f] - cand[f]).max():.3g} px')
for n in ('S2_h', 'S2_v'):
    rowdiff(n)
for f in ('Fx', 'Fy'):
    print(f'{f:6s} max |diff| {np.abs(ref[f] - cand[f]).max():.3g} px')
e = np.hypot(ref['mx4'] - cand['map_x'], ref['my4'] - cand['map_y'])
print(f'map4   max endpoint error {e.max():.4g} px, p99 {np.percentile(e, 99):.4g} px')
