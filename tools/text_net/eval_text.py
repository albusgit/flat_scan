"""After train_text.py: ONNX export, scores against the shipped rule net, overlay sheets on real pages, a report.
  python eval_text.py RUN_DIR VAL_TEXT VAL_RULE OUT_DIR
OUT_DIR: textnet.onnx (input 'page' 1x3xHxW BGR [0, 1], output 'lines' 1x4xHxW probabilities [rule h, rule v, text h,
text v], H, W multiples of 32), report.md, sheets/*.jpg (red / blue: rules, green / yellow: text lines, > 0.5).
Scores: F1 per channel at 0.5 on the held-out pools' centre 384 crops (as train_text.py), text-line F1 per script, and
the shipped rule net (app/src/main/assets/models/rulenet.onnx, ONNX Runtime) on the same rule crops."""
import glob, json, os, sys
import numpy as np, torch, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'rule_net'))
from model import RuleNet   # noqa: E402
import train_text as tt   # noqa: E402

run, vtext, vrule, out = sys.argv[1:5]
os.makedirs(os.path.join(out, 'sheets'), exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
tt.keep_awake()
ck = torch.load(os.path.join(run, 'best.pt'), map_location='cpu')
net = RuleNet(tuple(ck['ch']), 4).eval()
net.load_state_dict(ck['state'])
lines = [f"# Text-line net: run {os.path.basename(run)}", '',
         f"Checkpoint best.pt: iteration {ck['it']}, widths {ck['ch']}, {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M params.", '']


class Probs(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n

    def forward(self, x):
        return torch.sigmoid(self.n(x))


onnx_path = os.path.join(out, 'textnet.onnx')
torch.onnx.export(Probs(net).eval(), torch.rand(1, 3, 576, 416), onnx_path, input_names=['page'], output_names=['lines'], opset_version=17,
                  dynamic_axes={'page': {2: 'h', 3: 'w'}, 'lines': {2: 'h', 3: 'w'}}, do_constant_folding=True, dynamo=False)
lines.append(f'ONNX: {onnx_path} ({os.path.getsize(onnx_path) // 1024} KB, fp32).')

net = net.eval().cuda()   # the export leaves the wrapped net in training mode (Probs is made in it)
names = ['rule h', 'rule v', 'text h', 'text v']
lines += ['', '## Held-out synthetic pools (F1 at 0.5, centre 384 px crops)', '', '| pool | net | ' + ' | '.join(names) + ' |', '|---|---|---|---|---|---|']
Xt, Tt = tt.val_set(vtext, 100000)
Xr, Tr = tt.val_set(vrule, 100000)
ct = tt.f1_counts(net, Xt, Tt)
cr = tt.f1_counts(net, Xr, Tr)
fmt = lambda v: ' | '.join(f'{x:.4f}' for x in v)
lines.append(f'| text pages ({len(Xt)}) | new | {fmt(tt.f1(ct.sum(0)))} |')
lines.append(f'| rule pages, synth2 ({len(Xr)}) | new | {fmt(tt.f1(cr.sum(0)))} | - | - |')

# the shipped rule net on the same crops
import onnxruntime as ort   # noqa: E402
s = ort.InferenceSession(os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models', 'rulenet.onnx'), providers=['CPUExecutionProvider'])


def old_counts(X, T):
    res = []
    for a in range(0, len(X), 8):
        x = X[a:a + 8].astype(np.float32).transpose(0, 3, 1, 2) / 255
        p = np.concatenate([s.run(None, {'page': x[j:j + 1]})[0] for j in range(len(x))]) > 0.5   # its batch is 1
        y = T[a:a + 8, :2]
        res.append(np.stack([(p & y).sum((2, 3)), (p & ~y).sum((2, 3)), (~p & y).sum((2, 3))], -1))
    return np.concatenate(res)


lines.append(f'| text pages | shipped rule net | {fmt(tt.f1(old_counts(Xt, Tt).sum(0)))} | - | - |')
lines.append(f'| rule pages, synth2 | shipped rule net | {fmt(tt.f1(old_counts(Xr, Tr).sum(0)))} | - | - |')

scripts = json.load(open(vtext + '_script.json'))[:len(Xt)]
lines += ['', '## Text-line F1 per script (held-out text pages, h and v together)', '', '| script | pages | text F1 | rule F1 |', '|---|---|---|---|']
for sc in sorted(set(scripts), key=lambda q: -scripts.count(q)):
    idx = [i for i, q in enumerate(scripts) if q == sc]
    c = ct[idx].sum(0)
    tf = tt.f1(c[2:].sum(0)); rf = tt.f1(c[:2].sum(0))
    lines.append(f'| {sc} | {len(idx)} | {tf:.4f} | {rf:.4f} |')


def overlay(img_bgr, name):
    h, w = img_bgr.shape[:2]
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    sm = cv2.resize(img_bgr, (w2, h2), interpolation=cv2.INTER_AREA)
    ph, pw = (h2 + 31) // 32 * 32, (w2 + 31) // 32 * 32
    pad = cv2.copyMakeBorder(sm, 0, ph - h2, 0, pw - w2, cv2.BORDER_REPLICATE)
    x = torch.from_numpy(pad.astype(np.float32).transpose(2, 0, 1)[None] / 255).cuda()
    with torch.no_grad():
        p = torch.sigmoid(net(x)).float().cpu().numpy()[0, :, :h2, :w2] > 0.5
    v = sm.copy()
    v[p[2]] = (0, 200, 0); v[p[3]] = (0, 220, 220); v[p[0]] = (0, 0, 255); v[p[1]] = (255, 0, 0)
    cv2.imwrite(os.path.join(out, 'sheets', name + '.jpg'), np.hstack([sm, v]), [cv2.IMWRITE_JPEG_QUALITY, 80])


done = 0
for f in sorted(glob.glob('E:/fs_datasets/runs/sp_new/real35/pages/*.jpg') + glob.glob('E:/fs_datasets/runs/sp_new/real35/pages/*.png'))[:80]:
    if '.lines.' in f:
        continue
    im = cv2.imread(f)
    if im is not None:
        overlay(im, 'real35_' + os.path.splitext(os.path.basename(f))[0]); done += 1
for sset in ('wildreceipt', 'funsd', 'cord', 'commons_colour_docs'):
    try:
        items = json.load(open(f'E:/fs_datasets/sets/{sset}.json'))['items'][:6]
    except Exception:
        continue
    for it in items:
        im = cv2.imread(it['photo'])
        if im is None:
            continue
        k = 1600 / max(im.shape[:2])
        if k < 1:
            im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        overlay(im, f'{sset}_{it["id"]}'); done += 1
lines += ['', f'Overlay sheets: {done} in {os.path.join(out, "sheets")} (the flattened real35 pages of run sp_new, and photos of',
          'wildreceipt, funsd, cord, commons_colour_docs as they are, not flattened).', '',
          'Training log: ' + os.path.join(run, 'log.jsonl')]
open(os.path.join(out, 'report.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines), flush=True)
