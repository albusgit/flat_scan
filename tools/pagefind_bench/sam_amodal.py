"""sam_amodal.py IN.npz OUT.npz [--show SHEET.jpg]: SAM outlines (sam_pack.py, sources NAME_sam) made whole where something
lies over the paper. SAM outlines the paper it sees: a thumb over the page's edge leaves a notch in its mask, a pen on
the page a hole. The page finder's target is the whole sheet (the generator draws thumbs, hands and pens over pages and
keeps the whole sheet as the target: gpugen/occluders.py), so these are filled:
  - holes inside the paper: always;
  - notches into the outline (gaps between the mask and its convex hull) deeper than they are wide (depth >= 0.7 x the
    opening along the hull, and >= 3 px at 384): a finger or a thumb. Wide shallow gaps stay (a curved or bowed side,
    a book's dipping top edge), and so do triangles at a corner (a dog-ear: as deep as half its opening).
Other sources (4-corner fills) are copied as they are."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train', 'pagefind2'))
import data as D  # noqa: E402

SIZE = 384


def make_whole(m, min_depth=3.0, ratio=0.7):
    """m: H x W bool -> (filled mask, number of holes filled, number of notches filled)"""
    m8 = m.astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return m, 0, 0
    hull = np.zeros_like(m8)
    cv2.fillPoly(hull, [cv2.convexHull(max(cs, key=cv2.contourArea))], 1)
    gap = (hull > 0) & ~m
    n, lab = cv2.connectedComponents(gap.astype(np.uint8), connectivity=4)
    if n <= 1:
        return m, 0, 0
    depth = cv2.distanceTransform(hull, cv2.DIST_L2, 3)                 # px to the hull's outline
    rim = cv2.dilate((hull == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out, holes, notches = m.copy(), 0, 0
    for i in range(1, n):
        comp = lab == i
        opening = int((comp & rim).sum())
        if opening == 0:
            out |= comp
            holes += 1
            continue
        d = float(depth[comp].max())
        if d >= min_depth and d >= ratio * opening:
            out |= comp
            notches += 1
    return out, holes, notches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--show', default='')
    a = ap.parse_args()
    z = D.load(a.inp)
    srcs = list(z['sources'])
    masks = z['mask'].copy()
    changed, st = [], {'holes': 0, 'notches': 0, 'samples': 0}
    for k in range(len(masks)):
        if not srcs[z['src'][k]].endswith('_sam'):
            continue
        m = np.unpackbits(masks[k])[:SIZE * SIZE].reshape(SIZE, SIZE).astype(bool)
        m2, h, nt = make_whole(m)
        if h or nt:
            masks[k] = np.packbits(m2.ravel())
            st['holes'] += h
            st['notches'] += nt
            st['samples'] += 1
            changed.append((k, m, m2))
    z['mask'] = masks
    np.savez(a.out, **z)
    print(f"{len(masks)} samples, {sum(s.endswith('_sam') for s in np.array(srcs)[z['src']])} SAM outlines; made whole: {st}")
    if a.show and changed:
        tiles = []
        rng = np.random.default_rng(0)
        for k, m, m2 in [changed[i] for i in rng.permutation(len(changed))[:24]]:
            im = cv2.imdecode(z['jpeg'][z['off'][k]:z['off'][k + 1]], cv2.IMREAD_COLOR)
            for mm, col in ((m, (40, 220, 40)), (m2, (230, 60, 230))):
                cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, col, 1)
            add = m2 & ~m
            im[add] = (0.5 * im[add] + [115, 30, 115]).astype(np.uint8)
            cv2.putText(im, str(z['name'][k])[:40], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            tiles.append(im)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(a.show, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print('wrote', a.show)


if __name__ == '__main__':
    main()
