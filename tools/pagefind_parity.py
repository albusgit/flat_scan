"""App page finder (core/src/pagefind.cpp via tools/desktop_check pagefind_cli) vs the reference seg/pagefind.py with
PAGE_MODEL, on the real photos of the round-4 pack.

  python tools/pagefind_parity.py PACK_DIR MODEL.onnx PAGEFIND_CLI OUTDIR [PHOTO_DIR]
  (PHOTO_DIR: other photos, e.g. synthetic book spreads; default PACK_DIR/real_photos)

Per photo:
  1. net input: pagefind_cli input (C++ segnet.to_input) vs segnet.to_input
  2. the net (onnxruntime, CPU) on the C++ input -> logits, as the app runs it (ONNX Runtime)
  3. pagefind_cli find (C++ post-processing) vs pagefind.paper_mask(img, Predictor(MODEL), return_info=True)
Prints app-vs-reference mask agreement and quality fields, and the IoU of both against the
hand-made GT (scored like seg/evalseg.py: 1000 px long side, nearest)."""
import json, os, subprocess, sys, types
import numpy as np, cv2

PACK, MODEL, CLI, OUT = sys.argv[1:5]
PHOTOS = sys.argv[5] if len(sys.argv) > 5 else os.path.join(PACK, 'real_photos')
os.makedirs(OUT, exist_ok=True)
try:
    import torch  # noqa: F401  (segnet imports it at module level; the ONNX path never uses it)
except ImportError:
    t, nn, F = types.ModuleType('torch'), types.ModuleType('torch.nn'), types.ModuleType('torch.nn.functional')
    nn.Module = object
    t.nn = nn
    sys.modules.update({'torch': t, 'torch.nn': nn, 'torch.nn.functional': F})
sys.path.insert(0, os.path.join(PACK, 'seg'))
import segnet, pagefind  # noqa: E402

GT = json.load(open(os.path.join(PACK, 'gt', 'gt.json')))
pred = segnet.Predictor(MODEL)
print('opencv', cv2.__version__, ' numpy', np.__version__, ' onnxruntime', __import__('onnxruntime').__version__)


def gt_iou(mask, name):
    g = GT.get(name)
    if not g or not g['page']:
        return None
    gt = (cv2.imread(os.path.join(PACK, 'gt', 'masks', f'{name}.png'), 0) > 127).astype(np.uint8)
    s = 1000 / max(gt.shape)
    gs = cv2.resize(gt, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    ps = cv2.resize(mask.astype(np.uint8), (gs.shape[1], gs.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (gs & ps).sum() / max(1, (gs | ps).sum())


def edge_dist(a, b):
    """mean and max distance (px, full resolution) from each mask's outline to the other's"""
    def edge(m):
        return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ea, eb = edge(a), edge(b)
    if not ea.any() or not eb.any():
        return np.nan, np.nan
    da = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)
    return 0.5 * (da[ea].mean() + db[eb].mean()), max(da[ea].max(), db[eb].max())


rows = []
names = sorted(f[:-4] for f in os.listdir(PHOTOS) if f.endswith('.jpg'))
for n in names:
    path = os.path.join(PHOTOS, n + '.jpg')
    img = cv2.imread(path)
    pre = os.path.join(OUT, n)
    subprocess.run([CLI, 'input', path, pre + '_x.npz'], check=True)
    x_cpp = np.load(pre + '_x.npz')['x']
    x_ref = segnet.to_input(img)
    dx = float(np.abs(x_cpp - x_ref).max())
    lg = pred.sess.run(None, {pred.sess.get_inputs()[0].name: x_cpp})[0][0].astype(np.float32)
    np.savez(pre + '_logits.npz', logits=lg)
    subprocess.run([CLI, 'find', path, pre + '_logits.npz', pre], check=True)
    app = json.load(open(pre + '.json'))
    m_app = (cv2.imread(pre + '_page0.png', 0) > 127).astype(np.uint8)
    m_ref, c_ref, info = pagefind.paper_mask(img, pred, return_info=True)
    m_ref = m_ref.astype(np.uint8)
    inter, union = (m_app & m_ref).sum(), (m_app | m_ref).sum()
    iou = inter / max(1, union)
    ndiff = int((m_app != m_ref).sum())
    mean_d, max_d = edge_dist(m_app, m_ref)
    ref_pages = info.get('pages', 1) if not info.get('fallback') else 1
    r = dict(name=n, dx=dx, iou_app_ref=iou, diff_px=ndiff, diff_frac=ndiff / m_ref.size, edge_mean=mean_d, edge_max=max_d,
             pages_app=app['pages'], pages_ref=ref_pages, fb_app=app['whole_frame'], fb_ref=info.get('fallback') == 'whole_frame',
             margin_app=app['net_margin'], margin_ref=info['net_margin'],
             strength_app=app['edge_strength'], strength_ref=info.get('edge_strength'),
             limit_app=app['edge_at_limit'], limit_ref=info.get('edge_at_limit'),
             gt_app=gt_iou(m_app, n), gt_ref=gt_iou(m_ref, n))
    rows.append(r)
    f = lambda v: 'n/a ' if v is None else f'{v:.4f}'
    print(f"{n:36s} input max|d| {dx:.1e}  app~ref IoU {iou:.6f}  diff px {ndiff:6d} ({100 * r['diff_frac']:.4f}%)  "
          f"edge mean/max {mean_d:.3f}/{max_d:.2f} px  pages {app['pages']}/{ref_pages}  fallback {int(r['fb_app'])}/{int(r['fb_ref'])}  "
          f"margin {app['net_margin']:.4f}/{info['net_margin']:.4f}  strength {f(app['edge_strength'] if app['edge_strength'] >= 0 else None)}/{f(info.get('edge_strength'))}  "
          f"GT IoU app {f(r['gt_app'])} ref {f(r['gt_ref'])}", flush=True)

I = [r['iou_app_ref'] for r in rows]
ga = np.array([r['gt_app'] for r in rows if r['gt_app'] is not None])
gr = np.array([r['gt_ref'] for r in rows if r['gt_ref'] is not None])
print(f"\n{len(rows)} photos. app vs reference mask IoU: min {min(I):.6f} mean {np.mean(I):.6f}; "
      f"max differing pixels {max(r['diff_frac'] for r in rows) * 100:.4f}% of the photo; "
      f"max outline distance {np.nanmax([r['edge_max'] for r in rows]):.2f} px, worst mean {np.nanmax([r['edge_mean'] for r in rows]):.3f} px")
print(f"net input max |C++ - Python|: {max(r['dx'] for r in rows):.2e}")
print(f"page count mismatches: {sum(r['pages_app'] != r['pages_ref'] for r in rows)}; whole-frame fallback mismatches: {sum(r['fb_app'] != r['fb_ref'] for r in rows)}")
if len(ga):
    print(f"GT ({len(ga)} photos): app mean IoU {ga.mean():.4f}, IoU<0.9: {(ga < 0.9).sum()}  |  reference mean IoU {gr.mean():.4f}, IoU<0.9: {(gr < 0.9).sum()}")
json.dump(rows, open(os.path.join(OUT, 'parity.json'), 'w'), indent=1, default=float)
