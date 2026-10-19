"""Contact sheets: every filter on every page, plus 1:1 crops of the faintest and the most
coloured text found on each page (for the "never lose text" check by eye).
usage: python -I tools/contact_sheet.py OUT_DIR NAME=RENDER_DIR ..."""
import sys, cv2, numpy as np
NAMES = ['Original', 'Lighten', 'Auto color', 'Grayscale', 'High contrast', 'Black & white']
out = sys.argv[1]; pages = [a.split('=') for a in sys.argv[2:]]
def label(img, t):
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 28), (255, 255, 255), -1)
    cv2.putText(img, t, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA); return img
def window(mask, wh=(300, 200)):
    s = cv2.boxFilter(mask.astype(np.float32), -1, wh, normalize=False)
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return max(0, x - wh[0] // 2), max(0, y - wh[1] // 2)
rows, crops = [], []
for name, d in pages:
    ims = [cv2.imread(f'{d}/f{k}.png') for k in range(6)]
    h = 420
    rows.append(np.hstack([label(cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h), interpolation=cv2.INTER_AREA), f'{name}: {NAMES[k]}') for k, i in enumerate(ims)]))
    g = cv2.cvtColor(ims[0], cv2.COLOR_BGR2GRAY); c = cv2.medianBlur(g, 41).astype(int) - g
    hsv = cv2.cvtColor(ims[0], cv2.COLOR_BGR2HSV); sat, val = hsv[..., 1], hsv[..., 2]
    ink = (c >= 20).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 60), np.uint8)) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((60, 1), np.uint8))
    nolines = cv2.dilate(lines, np.ones((7, 7), np.uint8)) == 0
    faint = (c >= 20) & (c < 45) & (sat < 60) & nolines
    # coloured ink: the stroke pixels themselves are saturated and not dark (red / blue text)
    colour = (c >= 40) & (sat > 120) & (val > 90) & nolines
    for what, m in (('faint text', faint), ('coloured text', colour)):
        x, y = window(m)
        crops.append(np.hstack([label(i[y:y + 200, x:x + 300], f'{name} {what}: {NAMES[k]}' if k == 0 else NAMES[k]) for k, i in enumerate(ims)]))
w = max(r.shape[1] for r in rows)
cv2.imwrite(f'{out}/contact_sheet.jpg', np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0)), constant_values=200) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])
cv2.imwrite(f'{out}/text_crops.png', np.vstack([np.pad(r, ((0, 4), (0, 0), (0, 0)), constant_values=200) for r in crops]))
