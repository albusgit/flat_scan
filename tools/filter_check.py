"""Filter acceptance: text that is readable in Original must stay readable in every filter.
usage: python -I tools/filter_check.py RENDER_DIR... (each holds f0.png .. f5.png from `fs_cli render`)
Local paper level: 41-px median of the grey image (text covers far less than half of the
window). Ink in Original: >= 20 levels darker than the local paper. Retained in a filter:
>= 8 levels darker than that filter's local paper. (A dilated, local-maximum background
counts paper grain of phone photos as ink, so it is not used here.)
A mark (8-connected ink component, >= 3 px) is lost if none of its pixels is retained."""
import sys, os, cv2, numpy as np
NAMES = ['Original', 'Lighten', 'Auto color', 'Grayscale', 'High contrast', 'Black & white']
def contrast(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, 41)
    return bg.astype(np.int16) - g
worst = 0
for d in sys.argv[1:]:
    o = cv2.imread(f'{d}/f0.png')
    c0 = contrast(o)
    ink = c0 >= 20
    hsv = cv2.cvtColor(o, cv2.COLOR_BGR2HSV)
    faint = ink & (c0 < 60)
    colour = ink & (hsv[..., 1] > 70)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    big = st[:, 4] >= 3; big[0] = False
    print(f'{os.path.basename(os.path.dirname(d))}: ink px {int(ink.sum())}, faint {int(faint.sum())}, coloured {int(colour.sum())}, marks {int(big.sum())}')
    for k in range(1, 6):
        f = cv2.imread(f'{d}/f{k}.png')
        kept = contrast(f) >= 8
        keep_lab = np.zeros(n, bool)
        keep_lab[np.unique(lab[kept & ink])] = True
        lost = int((big & ~keep_lab).sum())
        worst = max(worst, lost)
        r = lambda m: 100.0 * (kept & m).sum() / max(1, m.sum())
        print(f'   {NAMES[k]:14s} ink retained {r(ink):6.2f}%  faint {r(faint):6.2f}%  coloured {r(colour):6.2f}%  marks lost {lost}')
print('worst marks lost in any filter/page:', worst)
