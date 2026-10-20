"""Shading-map error on held-out synthetic pages (synth.sample, fixed seed; the training runs never
see these seeds' exact pages), for ShadeNet ONNX models and GC-Net (GCDRNet's illumination net).
  python synth_bench.py N NAME=MODEL.onnx ... [gc=GCNET.onnx]
Per page: ratio r = predicted map / true map, scaled so its median over paper pixels is 1 (a map
may differ by a global level: the filters set white); error = mean |log r| (0.1 ~ 10% off) over
  content: coloured or dark page content (albedo saturation > 0.25 or luma < 0.45) - error here
           is colour or ink bleached / darkened as if it were shading;
  shadow:  pixels whose true map is < 0.8 of the lit level - error here is shadow left in;
  all:     every pixel."""
import os, random, sys
import numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth

N = int(sys.argv[1])
models = dict(a.split('=', 1) for a in sys.argv[2:])
sess = {k: ort.InferenceSession(v, providers=['CPUExecutionProvider']) for k, v in models.items()}
rng = random.Random(987654321)
err = {k: {'content': [], 'shadow': [], 'all': []} for k in models}
for i in range(N):
    h, w = 512, 384
    o, T, a, kinds = synth.sample(rng, h, w)
    lum_a = 0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]
    sat_a = (a.max(2) - a.min(2)) / np.maximum(a.max(2), 1e-3)
    content = (sat_a > 0.25) | (lum_a < 0.45)
    paper = ~content
    Tl = T.mean(2)
    shadow = Tl < 0.8
    x = o.transpose(2, 0, 1)[None].astype(np.float32)
    for k, s in sess.items():
        S = s.run(None, {s.get_inputs()[0].name: x})[0][0].transpose(1, 2, 0)
        S = np.maximum(S, 0.02)
        r = np.log(S.mean(2) / np.maximum(Tl, 0.02))
        r -= np.median(r[paper]) if paper.sum() > 100 else np.median(r)
        r = np.abs(r)
        err[k]['all'].append(r.mean())
        if content.sum() > 200: err[k]['content'].append(r[content].mean())
        if shadow.sum() > 200: err[k]['shadow'].append(r[shadow].mean())
for k in models:
    e = err[k]
    print(f"{k:12s} content {np.mean(e['content']):.3f} (n={len(e['content'])}, p90 {np.percentile(e['content'], 90):.3f})  "
          f"shadow {np.mean(e['shadow']):.3f} (n={len(e['shadow'])}, p90 {np.percentile(e['shadow'], 90):.3f})  all {np.mean(e['all']):.3f}")
