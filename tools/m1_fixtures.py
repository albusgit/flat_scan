"""Milestone-1 fixtures: the reference resample of each page from its saved 4-px-grid map.
usage: python -I tools/m1_fixtures.py REF_CODE_DIR REF_OUT_DIR TEST_IMAGES_DIR FIXTURE_DIR"""
import sys, os
code, out, imgs, fx = sys.argv[1:5]
sys.path.insert(0, os.path.abspath(code))
import cv2, numpy as np, warpmap as W
pages = {'photo1': 'photo1.jpg', 'photo2': 'photo2.jpg', 'scan_p1': 'scan_p1.jpg', 'scan_p2': 'scan_p2.jpg', 'scan_p3': 'scan_p3.jpg'}
for name, src in pages.items():
    os.makedirs(f'{fx}/{name}', exist_ok=True)
    m = W.load_maps(f'{out}/{name}/warpmap.npz')
    img = cv2.imread(f'{imgs}/{src}')
    ref = cv2.remap(img, m['map_x'], m['map_y'], cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    cv2.imwrite(f'{fx}/{name}/m1_resample_ref.png', ref)
    # how far the reference's own published image is from its stored grid map
    pub = cv2.imread(f'{out}/{name}/' + ('out.png' if name.startswith('photo') else 'B_warpmap.png'))
    d = np.abs(pub.astype(int) - ref)
    print(f'{name}: grid-expanded resample vs published reference image: max {d.max()}, pixels>1 {(d > 1).sum()}')
