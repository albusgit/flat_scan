"""eval.py MODEL.onnx LIST... [--margin M] [--turns]: orientnet on real pages.

LIST: a text file of page images (one path per line) or a folder (*.jpg, *.png). With --turns every page is assumed
upright and tried in all 4 turns (a page that is upright tested 4 times); otherwise each page is classified as it is
and the answer printed (for pages whose true turn is not known: look at them).
Per margin threshold: accuracy of the pages decided (margin >= M), the share abstained (left as they are) and the
share turned WRONG - the number that matters: a wrong turn is worse than no turn.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

import orientnet as on


def run_page(sess, gray):
    p = on.page_patches(gray)
    if len(p) == 0:
        return None
    lg = sess.run(None, {'patches': on.to_input(p)})[0]
    lp = lg - np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True)) - lg.max(1, keepdims=True)
    return lp.mean(0), len(p)


def files_of(arg):
    if os.path.isdir(arg):
        return sorted(glob.glob(os.path.join(arg, '*.jpg')) + glob.glob(os.path.join(arg, '*.png')))
    return [l.strip() for l in open(arg, encoding='utf8') if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('lists', nargs='+')
    ap.add_argument('--turns', action='store_true')
    ap.add_argument('--margins', default='0,0.5,1,1.5,2,3')
    ap.add_argument('--show', type=int, default=0, help='print the N least sure / wrong pages')
    a = ap.parse_args()
    sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
    rows = []
    for L in a.lists:
        for f in files_of(L):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            turns = range(4) if a.turns else [None]
            for k in turns:
                # the page turned k x 90 ccw: the right answer is k
                gk = np.ascontiguousarray(np.rot90(g, k)) if k else g
                r = run_page(sess, gk)
                if r is None:
                    rows.append((f, k, None, 0.0, 0))
                    continue
                lp, n = r
                pred, margin = on.decide(lp, 0.0)
                rows.append((f, k, pred, margin, n))
            if not a.turns:
                f_, k_, p_, m_, n_ = rows[-1]
                print(f'{os.path.basename(f)[:50]:50s} turn {p_} margin {m_:.2f} patches {n_}')
    if a.turns:
        print(f'{len(rows)} tests ({len(rows) // 4} pages x 4 turns); no text patches: {sum(r[2] is None for r in rows)}')
        for M in [float(x) for x in a.margins.split(',')]:
            dec = [r for r in rows if r[2] is not None and r[3] >= M]
            right = sum(r[2] == r[1] for r in dec)
            wrong = len(dec) - right
            # abstaining leaves the page as it is: right only for k = 0
            print(f'margin >= {M:.1f}: decided {len(dec) / len(rows):.3f}, of them right {right / max(1, len(dec)):.4f}; '
                  f'wrong turns {wrong} ({wrong / len(rows):.4f} of all tests)')
        if a.show:
            bad = sorted([r for r in rows if r[2] is not None and r[2] != r[1]], key=lambda r: -r[3])[:a.show]
            for r in bad:
                print('  wrong:', os.path.basename(r[0]), 'true', r[1], 'said', r[2], f'margin {r[3]:.2f} patches {r[4]}')


if __name__ == '__main__':
    main()
