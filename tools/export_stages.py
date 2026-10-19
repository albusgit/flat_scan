"""Per-stage fixtures from the Python reference (milestone 2 parity checks).

usage: python -I tools/export_stages.py REF_CODE_DIR photo|scan IMAGE OUTDIR [--mode surface|v8]

Re-runs the reference step by step in exactly the order of photo_pipeline.run (mode surface
or v8) / warpmap.py __main__, using the reference functions themselves, and saves every
intermediate the port is compared against. Nothing here changes the algorithm; it only
unrolls warpmap.unwarp_image / local_field so their internal results can be saved.
"""
import sys, os, json, argparse
p = argparse.ArgumentParser()
p.add_argument('code'); p.add_argument('kind'); p.add_argument('image'); p.add_argument('out')
p.add_argument('--mode', default='surface')
a = p.parse_args()
sys.path.insert(0, os.path.abspath(a.code))
import numpy as np, cv2
import corners as C, surface as SF, warpmap as WM, orient as O
os.makedirs(a.out, exist_ok=True)
info = {}


def lines_arr(L):
    return np.array([list(map(float, l)) for l in L], np.float64).reshape(-1, 5)


def warp_stages(src, tag):
    """warpmap.unwarp_image(src) unrolled; returns mx4, my4 and saves the internals."""
    h, w = src.shape[:2]
    # --- global stage (global_grid), with its control points
    Hs, Vs, _, _ = WM.detect(src)
    srcp, dstp = WM.build_controls(Hs, Vs)
    n_raw = len(dstp)
    A_img, GX, GY, nctrl = WM.global_grid(src)
    D = WM.darkness_fast(A_img)
    H, V = WM.get_lines(D, lsd=WM.LSD)
    S1 = WM.collect(D, [list(l) for l in H], [list(l) for l in V])
    T, ntext = WM.text_rows(D, H, V)
    (Fx, Fy, gl), H2, V2, S = WM.local_field(D, use_text=True)
    # pass-1 fields: replay local_field's first iteration
    G_saved = WM.G
    WM.G = WM.G_COARSE
    gw, gh = w // WM.G + 2, h // WM.G + 2
    Lm = WM.regularizer(gw, gh)
    sh = np.r_[S1['h'], T]
    wts = np.r_[np.ones(len(S1['h'])), np.full(len(T), WM.TEXT_W)]
    Fy1 = WM.solve(np.c_[sh[:, 0], sh[:, 1]], sh[:, 2], gw, gh, Lm, wts)
    Fx1 = WM.solve(np.c_[S1['v'][:, 1], S1['v'][:, 0]], S1['v'][:, 2], gw, gh, Lm)
    WM.G = G_saved
    S2 = WM.collect(D, H, V, prior_dx=(Fx1, WM.G_COARSE), prior_dy=(Fy1, WM.G_COARSE))
    k = WM.MAP_STRIDE
    Y4, X4 = np.mgrid[0:h:k, 0:w:k].astype(np.float32)
    lx = X4 + WM.grid_sample(Fx, gl, X4, Y4); ly = Y4 + WM.grid_sample(Fy, gl, X4, Y4)
    mx4 = lx + WM.grid_sample(GX, WM.GLOBAL_G, lx, ly); my4 = ly + WM.grid_sample(GY, WM.GLOBAL_G, lx, ly)
    cv2.imwrite(f'{a.out}/{tag}_A_img.png', A_img)
    np.savez(f'{a.out}/{tag}_warp.npz',
             ctrl_src_raw=srcp, ctrl_dst_raw=dstp, GX=GX, GY=GY, D=D.astype(np.float32),
             H=lines_arr(H), V=lines_arr(V),
             S1_h=S1['h'], S1_v=S1['v'], T=T, Fx1=Fx1, Fy1=Fy1, S2_h=S2['h'], S2_v=S2['v'],
             Fx=Fx, Fy=Fy, mx4=mx4, my4=my4)
    info[tag] = dict(n_ctrl_raw=int(n_raw), n_ctrl=int(nctrl), n_h=len(H), n_v=len(V), n_text=int(ntext),
                     n_S1_h=len(S1['h']), n_S1_v=len(S1['v']), n_S2_h=len(S2['h']), n_S2_v=len(S2['v']), g_local=int(gl))
    return mx4, my4


img = cv2.imread(a.image)
if a.kind == 'scan':
    mx4, my4 = warp_stages(img, 'scan')
    h, w = img.shape[:2]
    np.savez(f'{a.out}/final.npz', stride=4, shape=np.array([h, w]), map_x=mx4, map_y=my4)
else:
    import photo_pipeline as PP
    mask, contour = C.paper_mask(img)
    corners, sides, fold_d, lab = C.curved_corners(contour)
    pts = [contour[lab == k] for k in range(4)]
    wpx = np.linalg.norm(corners[1] - corners[0]); hpx = np.linalg.norm(corners[3] - corners[0])
    landscape = wpx > hpx
    W, H = PP.OUT_W, int(round(PP.OUT_W * 297 / 210))
    if landscape:
        W, H = H, W
    np.savez(f'{a.out}/detect.npz', mask=mask, contour=contour, lab=lab, corners=corners,
             side_p=np.array([s.p for s in sides]), side_inl=np.array([s.inl.sum() for s in sides]), fold_d=fold_d)
    info.update(landscape=bool(landscape), W=W, H=H, mode=a.mode)
    # homography page (always: orientation per the brief, and the v8 global map)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    Minv = np.linalg.inv(M)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    q = cv2.perspectiveTransform(np.c_[xx.ravel(), yy.ravel()].reshape(-1, 1, 2), Minv)[:, 0, :]
    hx, hy = q[:, 0].reshape(H, W).astype(np.float32), q[:, 1].reshape(H, W).astype(np.float32)
    homo = cv2.remap(img, hx, hy, cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    if a.mode == 'v8':
        ux, uy = hx, hy
    else:
        m = SF.fit(pts, corners, img.shape, landscape=landscape)
        info.update(surface=dict(axis=m['axis'], c=list(map(float, m['c'])), rvec=list(map(float, m['rvec'])),
                                 tvec=list(map(float, m['tvec'])), f=m['f'], cost=m['cost'], median_px=m['median_px'], p90_px=m['p90_px']))
        ux, uy = SF.unroll_map(m, out_w=W)
    stage1 = cv2.remap(img, ux, uy, cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    cv2.imwrite(f'{a.out}/stage1.png', stage1)
    cv2.imwrite(f'{a.out}/homography.png', homo)
    # orientation exactly as photo_pipeline (on stage 1), and on the homography page (brief)
    def decide(page):
        g1 = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        sh, sv = O._text_axis_score(g1)
        if sv > sh:
            ccw = cv2.rotate(g1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            u = O._upright_hebrew_score(ccw)
            return (cv2.ROTATE_90_COUNTERCLOCKWISE if u >= 0 else cv2.ROTATE_90_CLOCKWISE), [float(sh), float(sv), float(u)]
        u = O._upright_hebrew_score(g1)
        return (cv2.ROTATE_180 if u < 0 else None), [float(sh), float(sv), float(u)]
    rot_s1, sc_s1 = decide(stage1)
    rot_h, sc_h = decide(homo)
    info.update(orient_stage1=dict(rot=rot_s1, scores=sc_s1), orient_homography=dict(rot=rot_h, scores=sc_h))
    mx4, my4 = warp_stages(stage1, 'stage1')
    k = WM.MAP_STRIDE
    mx = cv2.remap(mx4, xx / k, yy / k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    my = cv2.remap(my4, xx / k, yy / k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fx = cv2.remap(ux, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fy = cv2.remap(uy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = cv2.remap(img, fx, fy, cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    # composition on the coarse grid only (what the app does): unroll map sampled at mx4,my4
    np.savez(f'{a.out}/unroll.npz', ux4=ux[::k, ::k], uy4=uy[::k, ::k], hx4=hx[::k, ::k], hy4=hy[::k, ::k])
    np.savez(f'{a.out}/final_dense.npz', map_x=fx, map_y=fy)     # unrotated, dense (reference)
    np.savez(f'{a.out}/final.npz', stride=k, shape=np.array(fx.shape), map_x=fx[::k, ::k], map_y=fy[::k, ::k])
    cv2.imwrite(f'{a.out}/out_unrotated.png', out)
json.dump(info, open(f'{a.out}/info.json', 'w'), indent=1)
print(json.dumps(info))
