"""Solid colors acceptance (measured on the output image only).
usage: python -I tools/solid_check.py NAME=RENDER_DIR ...   (f2.png = Auto color, f2_solid.png = + Solid colors)
* flat colours: colours of pixels whose 5x5 neighbourhood is identical (filled background),
  with >= 2000 px. One entry per colour cluster + pure white means same-colour cells match exactly.
* white check: flat-ish near-white pixels (all channels 240..254) that are not pure white.
* uniform cells: 5x5-flat share of the non-ink pixels inside filled (non-white) areas."""
import sys, cv2, numpy as np
for arg in sys.argv[1:]:
    name, d = arg.split('=')
    o = cv2.imread(f'{d}/f2_solid.png')
    mn = cv2.erode(o, np.ones((5, 5), np.uint8)); mx = cv2.dilate(o, np.ones((5, 5), np.uint8))
    flat = (mn == mx).all(2)
    cols, cnt = np.unique(o[flat].reshape(-1, 3), axis=0, return_counts=True)
    big = [(tuple(int(v) for v in c[::-1]), int(n)) for c, n in zip(cols, cnt) if n >= 2000]
    big.sort(key=lambda q: -q[1])
    nearwhite = ((o >= 240) & (o <= 254)).all(2)
    smooth = (mx.astype(int) - mn).max(2) <= 3
    print(f'{name}: flat colours (RGB, px): {big}')
    print(f'   near-white but not white, in smooth areas: {int((nearwhite & smooth).sum())} px')
    # filled areas: pixels whose flat colour is one of the non-white big colours, dilated
    fill = np.zeros(o.shape[:2], bool)
    for c, n in big:
        if c != (255, 255, 255): fill |= (o == np.array(c[::-1], np.uint8)).all(2)
    area = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    g = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY); ink = cv2.dilate((cv2.medianBlur(g, 41).astype(int) - g > 20).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    sel = area & ~ink
    print(f'   filled areas: {int(area.sum())} px, of their non-ink pixels exactly the cluster colour: {100.0 * fill[sel].mean() if sel.any() else 0:.2f}%')
