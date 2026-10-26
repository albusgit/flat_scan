"""check_synth.py PAGES_DIR OUT.jpg [N] [SEED]: a contact sheet of synth.sample: per sample the photo with the true
page border, the photo unwarped by its true grid (must be the flat page), and the app's net input with the target's
border (the turned crops included). A wrong ground truth shows at once: a bent or cut page in the middle column."""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402


def border(g):
    return np.concatenate([g[:, 0, :].T, g[:, :, -1].T, g[:, -1, ::-1].T, g[:, ::-1, 0].T])


def unwarp(img, g, oh, ow):
    gx = cv2.resize(g[0], (ow, oh), interpolation=cv2.INTER_LINEAR)
    gy = cv2.resize(g[1], (ow, oh), interpolation=cv2.INTER_LINEAR)
    return cv2.remap(img, gx, gy, cv2.INTER_LINEAR)


def main():
    pages = sorted(glob.glob(os.path.join(sys.argv[1], '*.png')) + glob.glob(os.path.join(sys.argv[1], '*.jpg')))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    kinds = ['flat', 'nearflat', 'curl', 'corner', 'fold', 'fold2', 'curl_corner', 'wave_curl']
    rows = []
    for i in range(n):
        sm = None
        while sm is None:
            sm = synth.sample(rng, pages, kinds[i % len(kinds)])
        ph = sm['photo'].copy()
        cv2.polylines(ph, [border(sm['grid']).astype(np.int32)], True, (0, 0, 255), 3)
        hw = sm['hw']
        ow = 600
        flat = unwarp(sm['photo'], sm['grid'], int(ow * hw), ow)
        ai = synth.app_input(sm, rng)
        x, t, turned = ai
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        tp = np.stack([(t[0] + 1) / 2 * (x.shape[1] - 1), (t[1] + 1) / 2 * (x.shape[0] - 1)])
        cv2.polylines(x, [border(tp).astype(np.int32)], True, (0, 200, 0), 3)
        # the net page as the app unwarps it from x
        netpage = unwarp(x, tp.astype(np.float32), 600 if not turned else int(600 / hw), int(600 / max(hw, 1e-3)) if turned else 600)
        H = 500
        f = lambda im: cv2.resize(im, (max(1, int(im.shape[1] * H / im.shape[0])), H), interpolation=cv2.INTER_AREA)
        r = np.hstack([f(ph), np.zeros((H, 6, 3), np.uint8), f(flat), np.zeros((H, 6, 3), np.uint8), f(x),
                       np.zeros((H, 6, 3), np.uint8), f(netpage)])
        cv2.putText(r, f"{sm['kind']} {sm['aspect']} hw {hw:.2f} turned {turned}", (10, 30), 0, 0.9, (0, 0, 255), 2)
        rows.append(r)
    Wm = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], Wm - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    cv2.imwrite(sys.argv[2], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    main()
