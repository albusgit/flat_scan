"""Contact sheets for the filter eval: rows = pages, columns = (run, filter) outputs.

  python -I sheet.py OUT.jpg SETDIR --cols LABEL=RUNDIR:K[,LABEL=RUNDIR:K...] [--pages a,b,c] [--h 700]
        [--crop x0,y0,x1,y1 (page fractions)] [--input] [--gt]
"""
import os, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_filters import pages, arg  # noqa: E402


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (40, 40, 40), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    out, setdir = sys.argv[1], sys.argv[2]
    cols = [c.split('=', 1) for c in arg('--cols', '').split(',') if c]
    H = int(arg('--h', 700))
    sel = arg('--pages', '')
    crop = [float(v) for v in arg('--crop', '0,0,1,1').split(',')]
    rows = []
    for n, p, pa, g in pages(setdir):
        if sel and n not in sel.split(','):
            continue
        tiles = []
        srcs = []
        if '--input' in sys.argv:
            srcs.append(('input', p))
        if '--gt' in sys.argv and g:
            srcs.append(('GT', g))
        for lab_, spec in cols:
            d, k = spec.rsplit(':', 1)
            srcs.append((lab_, os.path.join(d, n, f'f{k}.png')))
        for lab_, f in srcs:
            im = cv2.imread(f)
            if im is None:
                continue
            h, w = im.shape[:2]
            im = im[int(crop[1] * h):int(crop[3] * h), int(crop[0] * w):int(crop[2] * w)]
            h, w = im.shape[:2]
            im = cv2.resize(im, (max(1, int(w * H / h)), H), interpolation=cv2.INTER_AREA)
            tiles.append(label(im, f'{n[:28]} {lab_}'))
        if tiles:
            rows.append(np.hstack([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in tiles]))
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, len(rows), 'rows')


if __name__ == '__main__':
    main()
