"""sam_label.py: the page's true (curved) outline from a 4-corner label, with SAM 2.1 (Apache-2.0, facebookresearch/sam2;
weights E:/fs_models/sam2/sam2.1_hiera_large.pt). For training the page finder on real photos whose labels are only
4 corners (a quad: exact corners, straight sides where the paper may bow).

label(predictor, bgr, quad, delta) -> (mask bool H x W or None, info):
  1. SAM on the whole photo, prompted with the quad's box, its 4 corners (positive) and points just outside each
     side's middle (negative), 3 candidate masks; the one agreeing best with the filled quad;
  2. per side, a second SAM pass on a crop around that side (2x the side's band): finer edges than SAM's 256 px mask
     on the whole photo;
  3. accepted only where it is consistent with the label: the outline within delta (share of the page diagonal) of
     the quad's outline, passing within delta / 2 of each corner, IoU with the quad >= 0.85. Else None (keep the
     4-corner label with the interval loss).

  python sam_label.py test LIST GT_DIR OUT_DIR       the 33 hand-checked photos: quad from the true mask -> SAM -> scores
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import score_one  # noqa: E402

CKPT = 'E:/fs_models/sam2/sam2.1_hiera_large.pt'
CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'


def predictor(dev='cuda'):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(CFG, CKPT, device=dev))


def quad_mask(shape, quad):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(quad)).astype(np.int32)], 1)
    return m > 0


def outline_dist(mask):
    """distance (px) of every pixel to the mask's outline"""
    m = mask.astype(np.uint8)
    e = m - cv2.erode(m, np.ones((3, 3), np.uint8))
    return cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5)


def largest(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    out = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # holes filled
    cv2.drawContours(out, cs, -1, 1, -1)
    return out > 0


def _prompt(pred, quad, box_pad, H, W):
    q = np.asarray(quad, np.float64)
    c = q.mean(0)
    pts, lab = [c], [1]
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        pts.append(a + 0.08 * (c - a)); lab.append(1)                     # just inside each corner
        mid = (a + b) / 2
        n = mid - c
        n /= np.linalg.norm(n) + 1e-9
        out = mid + n * box_pad
        if 0 <= out[0] < W and 0 <= out[1] < H:
            pts.append(out); lab.append(0)                                   # outside each side's middle
    box = np.array([q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()])
    box = box + np.array([-1, -1, 1, 1]) * box_pad * 0.5
    box = np.clip(box, 0, [W - 1, H - 1, W - 1, H - 1])
    return np.array(pts, np.float32), np.array(lab, np.int32), box


def sam_whole(pred, bgr, quad):
    H, W = bgr.shape[:2]
    diag = np.hypot(*np.ptp(np.asarray(quad), 0))
    pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pts, lab, box = _prompt(pred, quad, 0.06 * diag, H, W)
    masks, scores, _ = pred.predict(point_coords=pts, point_labels=lab, box=box, multimask_output=True)
    qm = quad_mask(bgr.shape, quad)
    best, bi = None, -1
    for m in masks:
        m = largest(m > 0)
        iou = (m & qm).sum() / max(1, (m | qm).sum())
        if iou > bi:
            best, bi = m, iou
    return best, bi


def refine_sides(pred, bgr, mask, quad, band):
    """per side: SAM on a crop around the side (its band x 2 each way), prompted by the current mask's inside / outside
    near that side; the crop's mask replaces the current one within the band"""
    H, W = bgr.shape[:2]
    q = np.asarray(quad, np.float64)
    out = mask.copy()
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        x0, y0 = np.floor(np.minimum(a, b) - 2 * band).astype(int)
        x1, y1 = np.ceil(np.maximum(a, b) + 2 * band).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            continue
        crop = bgr[y0:y1, x0:x1]
        cm = mask[y0:y1, x0:x1]
        # prompts: points well inside / well outside the current mask near the side
        d_in = cv2.distanceTransform(cm.astype(np.uint8), cv2.DIST_L2, 5)
        d_out = cv2.distanceTransform((~cm).astype(np.uint8), cv2.DIST_L2, 5)
        rng = np.random.default_rng(k)
        pts, lab = [], []
        for d, l in ((d_in, 1), (d_out, 0)):
            ys, xs = np.nonzero(d > band * 0.6)
            if len(xs):
                sel = rng.choice(len(xs), min(4, len(xs)), replace=False)
                pts += [[xs[i], ys[i]] for i in sel]; lab += [l] * len(sel)
        if 1 not in lab or 0 not in lab:
            continue
        pred.set_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        lowres = cv2.resize(np.where(cm, 8.0, -8.0).astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        m, s, _ = pred.predict(point_coords=np.array(pts, np.float32), point_labels=np.array(lab), mask_input=lowres[None],
                               multimask_output=False)
        m = m[0] > 0
        # within the side's band only
        sel = np.zeros_like(cm)
        cv2.line(sel.view(np.uint8), tuple(np.round(a - [x0, y0]).astype(int)), tuple(np.round(b - [x0, y0]).astype(int)), 1,
                 int(2 * band) | 1)
        sel = sel.astype(bool)
        region = out[y0:y1, x0:x1]
        region[sel] = m[sel]
    return largest(out)


def label(pred, bgr, quad, delta=0.03, sides=True):
    q = np.asarray(quad, np.float64)
    diag = float(np.hypot(*np.ptp(q, 0)))
    m, iou0 = sam_whole(pred, bgr, quad)
    info = dict(iou_whole=round(float(iou0), 4))
    if m is None:
        return None, dict(info, why='no mask')
    if sides:
        m = refine_sides(pred, bgr, m, quad, max(8.0, delta * diag))
    qm = quad_mask(bgr.shape, quad)
    iou = (m & qm).sum() / max(1, (m | qm).sum())
    dq = outline_dist(qm)
    e = m.astype(np.uint8) - cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
    H, W = m.shape
    e[:3] = 0; e[-3:] = 0; e[:, :3] = 0; e[:, -3:] = 0                     # the photo's own border: no outline
    far = float(np.percentile(dq[e > 0], 99)) / diag if (e > 0).any() else 0.0
    dm = outline_dist(m)
    corner = max(float(dm[int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))]) for x, y in q
                 if 0 <= x < W and 0 <= y < H) / diag if any(0 <= x < W and 0 <= y < H for x, y in q) else 0.0
    info.update(iou_quad=round(float(iou), 4), outline_p99=round(far, 4), corner_miss=round(corner, 4))
    ok = iou >= 0.85 and far <= delta and corner <= 0.75 * delta
    # far apart (the label or SAM is wrong, and which one cannot be told: a hand's bite, a wrong label's corner,
    # see-through paper): the photo is better left out than trained on either outline
    info['unsure'] = bool(iou < 0.8 or far > 3 * delta or corner > 2 * delta)
    info['accepted'] = bool(ok)
    info['raw'] = m                     # SAM's mask whatever the checks said (for tests)
    return (m if ok else None), info


def quad_from_mask(gm):
    cs, _ = cv2.findContours(gm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for f in np.linspace(0.005, 0.1, 40):
        ap = cv2.approxPolyDP(c, f * peri, True)
        if len(ap) == 4:
            return ap[:, 0].astype(np.float64)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)


def test(lst, gt_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dev = os.environ.get('SAM_DEV', 'cuda')
    pred = predictor(dev)
    rows = []
    for p in open(lst).read().split():
        n = os.path.splitext(os.path.basename(p))[0]
        im = cv2.imread(p)
        gm = cv2.imread(os.path.join(gt_dir, 'masks', n + '.png'), 0)
        if im is None or gm is None:
            continue
        if gm.shape != im.shape[:2]:
            gm = cv2.resize(gm, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        g = gm > 127
        q = quad_from_mask(g)
        r = dict(name=n)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            for tag, sides in (('whole', False), ('sides', True)):
                m, info = label(pred, im, q, delta=0.03, sides=sides)
                raw = info.pop('raw', None)
                mm = m if m is not None else quad_mask(im.shape, q)       # what training would use
                iou, bdm, b95, _ = score_one(_contour(mm), g)
                ri, rbm, rb95, _ = score_one(_contour(raw), g) if raw is not None else (0, 1, 1, 0)
                r[tag] = dict(info, iou=round(iou, 4), bd_mean=round(bdm, 5), bd_p95=round(b95, 5),
                              raw_iou=round(ri, 4), raw_bd_mean=round(rbm, 5), raw_bd_p95=round(rb95, 5))
                if tag == 'sides' and raw is not None:
                    mm = raw
        qi, qbm, qb95, _ = score_one(q, g)
        r['quad'] = dict(iou=round(qi, 4), bd_mean=round(qbm, 5), bd_p95=round(qb95, 5))
        rows.append(r)
        print(n, 'quad p95 %.4f | SAM whole raw p95 %.4f | SAM sides raw p95 %.4f acc %s' % (
            r['quad']['bd_p95'], r['whole']['raw_bd_p95'], r['sides']['raw_bd_p95'], r['sides']['accepted']),
            {k: r['sides'].get(k) for k in ('iou_quad', 'outline_p99', 'corner_miss')}, flush=True)
        if m is not None or True:
            _draw(im, g, q, mm, os.path.join(out_dir, n + '.jpg'))
    json.dump(rows, open(os.path.join(out_dir, 'results.json'), 'w'), indent=1)
    for tag in ('quad', 'whole', 'sides'):
        p95 = np.array([r[tag]['bd_p95'] for r in rows]); bm = np.array([r[tag]['bd_mean'] for r in rows])
        acc = sum(r[tag].get('accepted', True) for r in rows)
        print(f'{tag:6s} n {len(rows)}  as used: boundary mean {bm.mean():.4f}  p95 mean {p95.mean():.4f}  p95 > 1%: {(p95 > 0.01).sum()}  '
              f'accepted {acc}')
        if tag != 'quad':
            p95 = np.array([r[tag]['raw_bd_p95'] for r in rows]); bm = np.array([r[tag]['raw_bd_mean'] for r in rows])
            print(f'{"":6s} raw SAM: boundary mean {bm.mean():.4f}  p95 mean {p95.mean():.4f}  p95 > 1%: {(p95 > 0.01).sum()}')


def _contour(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea)[:, 0].astype(np.float64) if cs else None


def _draw(im, g, q, m, path):
    s = 900 / max(im.shape[:2])
    v = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for mask, col, th in ((g, (40, 200, 40), 3), (m, (230, 120, 20), 1)):
        cs, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(v, cs, -1, col, th, cv2.LINE_AA)
    cv2.polylines(v, [np.round(q * s).astype(np.int32)], True, (40, 40, 230), 1, cv2.LINE_AA)
    cv2.imwrite(path, v, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == '__main__' and sys.argv[1] == 'test':
    test(sys.argv[2], sys.argv[3], sys.argv[4])
