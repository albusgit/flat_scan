"""Before/after crops at 200% (nearest neighbour) for Solid colors: for every fill colour of a
page, the window where that colour and text meet most, plus text in a white cell.
usage: python -I tools/solid_crops.py OUT.png NAME=RENDER_DIR ..."""
import sys, cv2, numpy as np
out = sys.argv[1]; rows = []
def label(img, t):
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 22), (255, 255, 255), -1)
    cv2.putText(img, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA); return img
for arg in sys.argv[2:]:
    name, d = arg.split('=')
    a, b = cv2.imread(f'{d}/f2.png'), cv2.imread(f'{d}/f2_solid.png')
    mn = cv2.erode(b, np.ones((5, 5), np.uint8)); mx = cv2.dilate(b, np.ones((5, 5), np.uint8))
    flat = (mn == mx).all(2)
    cols, cnt = np.unique(b[flat].reshape(-1, 3), axis=0, return_counts=True)
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); ink = (cv2.medianBlur(g, 41).astype(int) - g > 40).astype(np.float32)
    for c, n in zip(cols, cnt):
        if n < 2000: continue
        m = (b == c).all(2).astype(np.float32)
        score = cv2.boxFilter(m, -1, (200, 60), normalize=True) * cv2.boxFilter(ink, -1, (200, 60), normalize=True)
        y, x = np.unravel_index(np.argmax(score), score.shape)
        y0, x0 = max(0, y - 30), max(0, x - 100)
        ca, cb = a[y0:y0 + 60, x0:x0 + 200], b[y0:y0 + 60, x0:x0 + 200]
        up = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        what = 'white cell' if tuple(c) == (255, 255, 255) else f'fill RGB{tuple(int(v) for v in c[::-1])}'
        rows.append(np.hstack([label(up(ca), f'{name} {what}: before'), np.full((120, 6, 3), 128, np.uint8), label(up(cb), 'after (Solid colors)')]))
cv2.imwrite(out, np.vstack([np.pad(r, ((0, 6), (0, 0), (0, 0)), constant_values=128) for r in rows]))
print(len(rows), 'crops')
