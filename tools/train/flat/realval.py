"""Real photos with a flat ground-truth scan (dir300, DocReal: validation sets in E:/fs_datasets/sets),
flattened as the app runs UVDoc: pass 1 on the whole photo -> the page's corners -> the app's crop (box +
6 %, turned when landscape, core/src/uvdoc.cpp uv_crop / uv_input) -> pass 2 -> the page unwarped.

Scores, at the DocUNet protocol's 598400 px, grey:
  ms_ssim  multi-scale SSIM with the scan (higher is better)
  ld       mean DIS optical-flow length from the scan to the output, px (a stand-in for the DocUNet LD,
           which needs the SIFT-flow scorer; it ranks the same way, lower is better)

  load(set_json, every, root) -> items
  score(items, grid_fn, dev) -> dict     grid_fn(x float32 1x3x712x488 in [0, 1]) -> grid 2 x 45 x 31 in [-1, 1]
  python realval.py ONNX [SET] [EVERY]   the score of a UVDoc model file (e.g. the shipped one)"""
import json, os, sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IN_H, IN_W = 712, 488
AREA = 598400


def load(set_json, every=3, root=None):
    s = json.load(open(set_json))
    out = []
    for it in s['items'][::every]:
        p, g = it['photo'], it['gt']
        if root:   # the set files hold E:/fs_datasets paths
            p, g = p.replace('E:/fs_datasets', root), g.replace('E:/fs_datasets', root)
        a, b = cv2.imread(p), cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        s_ = min(1.0, 2000 / max(a.shape[:2]))
        if s_ < 1:
            a = cv2.resize(a, (round(a.shape[1] * s_), round(a.shape[0] * s_)), interpolation=cv2.INTER_AREA)
        k = np.sqrt(AREA / b.size)
        out.append((it['id'], a, cv2.resize(b, (round(b.shape[1] * k), round(b.shape[0] * k)), interpolation=cv2.INTER_AREA)))
    return out


def net_input(img):
    rgb = cv2.cvtColor(cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None]


def to_px(g, w, h):
    """[-1, 1] grid -> px of an image w x h (UVDoc's normalisation: (x - left) / width * 2 - 1)"""
    return (g[0] + 1) / 2 * w, (g[1] + 1) / 2 * h


def two_pass(img, grid_fn):
    """the page's backward grid in photo px (2 x 45 x 31)"""
    h, w = img.shape[:2]
    gx, gy = to_px(grid_fn(net_input(img)), w, h)
    c = np.array([[gx[0, 0], gy[0, 0]], [gx[0, -1], gy[0, -1]], [gx[-1, -1], gy[-1, -1]], [gx[-1, 0], gy[-1, 0]]])
    x0, y0 = c.min(0); x1, y1 = c.max(0)
    m = 0.06 * max(x1 - x0, y1 - y0)
    bx0, by0 = max(0, int(np.floor(x0 - m))), max(0, int(np.floor(y0 - m)))
    bx1, by1 = min(w, int(np.ceil(x1 + m)) + 1), min(h, int(np.ceil(y1 + m)) + 1)
    if bx1 - bx0 < 32 or by1 - by0 < 32:
        return np.stack([gx, gy])
    turned = np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[3] - c[0])
    crop = img[by0:by1, bx0:bx1]
    if turned:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    cw, ch = crop.shape[1], crop.shape[0]
    px, py = to_px(grid_fn(net_input(crop)), cw, ch)
    if turned:   # a point (u, v) of the turned crop is (v, ch0 - 1 - u) of the crop
        px, py = py, (by1 - by0 - 1) - px
    return np.stack([px + bx0, py + by0])


def unwarp(img, grid, out_hw):
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    h, w = img.shape[:2]
    g = torch.from_numpy(np.stack([grid[0] / (w - 1) * 2 - 1, grid[1] / (h - 1) * 2 - 1], -1)[None]).float()
    g = F.interpolate(g.permute(0, 3, 1, 2), size=out_hw, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
    return F.grid_sample(t, g, mode='bilinear', align_corners=True)[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()


def _gauss(x, w):
    c = x.shape[1]
    return F.conv2d(x, w.expand(c, 1, -1, -1), groups=c)


def ms_ssim(a, b):
    """5-scale MS-SSIM of two grey images (uint8, same size), window 11 / sigma 1.5 (Wang et al. 2003)"""
    x = torch.from_numpy(a).float()[None, None]; y = torch.from_numpy(b).float()[None, None]
    r = torch.arange(11).float() - 5
    g = torch.exp(-r ** 2 / (2 * 1.5 ** 2)); g = (g / g.sum())
    w = (g[:, None] * g[None, :])[None, None]
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    wts = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    vals = []
    for k in range(5):
        mx, my = _gauss(x, w), _gauss(y, w)
        sxx, syy, sxy = _gauss(x * x, w) - mx ** 2, _gauss(y * y, w) - my ** 2, _gauss(x * y, w) - mx * my
        cs = ((2 * sxy + C2) / (sxx + syy + C2)).mean()
        if k == 4:
            vals.append(((2 * mx * my + C1) / (mx ** 2 + my ** 2 + C1) * (2 * sxy + C2) / (sxx + syy + C2)).mean())
        else:
            vals.append(cs)
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    v = torch.stack(vals).clamp(min=1e-6)
    return float(torch.prod(v ** wts))


def ld(ref, out):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(ref, out, None)
    return float(np.sqrt((f ** 2).sum(-1)).mean())


def score(items, grid_fn, threads=8):
    """the grids on the GPU one by one (fast), the unwarp + MS-SSIM + flow (CPU) in threads (cv2 and torch release the GIL)"""
    from concurrent.futures import ThreadPoolExecutor
    grids = [two_pass(img, grid_fn) for _, img, _ in items]
    torch.set_num_threads(1)

    def one(k):
        _, img, ref = items[k]
        out = cv2.cvtColor(unwarp(img, grids[k], ref.shape), cv2.COLOR_BGR2GRAY)
        return ms_ssim(ref, out), ld(ref, out)
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(one, range(len(items))))
    return {'ms_ssim': float(np.mean([r[0] for r in res])), 'ld': float(np.mean([r[1] for r in res]))}


if __name__ == '__main__':
    import onnxruntime as ort
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from fast import data_root
    sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
    root = data_root()
    sj = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, 'sets', 'dir300.json')
    items = load(sj, int(sys.argv[3]) if len(sys.argv) > 3 else 3, root)
    print(len(items), 'photos')
    print(os.path.basename(sys.argv[1]), score(items, lambda x: sess.run(None, {'image': x})[0][0]))
