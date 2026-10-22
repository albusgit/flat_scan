"""ShadeNet inputs at every width the pipeline feeds, from real flattened pages, for compare.py --pages.

  python tools/litert/shade_inputs.py PAGES_DIR OUT.npz
      PAGES_DIR: *.jpg flattened pages (e.g. tools/accel/net_inputs.py's OUT_DIR, whose NAME.jpg are the 35 photos'
      pipeline outputs); OUT.npz: NAME@W -> uint8 BGR 512 x W x 3, W = 32, 64, .., 512
Run it with an environment that has opencv-python (the round4 venv). As core/src/pipeline.cpp apply_shade_net makes
the net input: portrait (a landscape page turned clockwise), INTER_AREA to 512 x W; compare.py divides by 255. The
page is stretched to each width (any width occurs for some page shape; the content stays a real page's)."""
import glob
import os
import sys

import cv2
import numpy as np

WIDTHS = list(range(32, 513, 32))


def main():
    src, out = sys.argv[1:3]
    d = {}
    for p in sorted(glob.glob(os.path.join(src, '*.jpg'))):
        if p.endswith('.lines.jpg'):
            continue
        im = cv2.imread(p)
        if im.shape[1] > im.shape[0]:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        for w in WIDTHS:
            d[f'{os.path.basename(p)[:-4]}@{w}'] = cv2.resize(im, (w, 512), interpolation=cv2.INTER_AREA)
    np.savez_compressed(out, **d)
    print(len(d), 'inputs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
