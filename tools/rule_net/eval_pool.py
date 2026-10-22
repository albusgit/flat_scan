"""Scores rule-net ONNX models on held-out pools (384-px centre crops, threshold 0.5):
F1 / precision / recall of each mask for the rules alone (pixels on a straight edge of a picture or dark
bar not counted either way) and with those edges counted as rules.
  python eval_pool.py POOL[,POOL...] MODEL.onnx [MODEL.onnx ...]"""
import os, sys
import numpy as np, onnxruntime as ort

pools, models = sys.argv[1].split(','), sys.argv[2:]
for pp in pools:
    X = np.load(pp + '_img.npy'); Y = np.load(pp + '_mask.npy')
    E = np.load(pp + '_edge.npy') if os.path.exists(pp + '_edge.npy') else np.zeros((len(X),) + X.shape[1:3], np.uint8)
    o = (X.shape[1] - 384) // 2
    X, Y, E = X[:, o:o + 384, o:o + 384], Y[:, :, o:o + 384, o:o + 384] > 127, E[:, o:o + 384, o:o + 384]
    Em = np.stack([(E & 1) > 0, (E & 2) > 0], 1)
    for m in models:
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        s = ort.InferenceSession(m, so, providers=['CPUExecutionProvider'])
        st = {k: np.zeros((3, 2)) for k in ('rules', 'with_edges')}
        for k in range(0, len(X), 8):
            x = X[k:k + 8].astype(np.float32).transpose(0, 3, 1, 2) / 255
            p = np.concatenate([s.run(None, {'page': x[i:i + 1]})[0] for i in range(len(x))]) > 0.5
            y, e = Y[k:k + 8], Em[k:k + 8]
            for name, yy, care in (('rules', y, ~(e & ~y)), ('with_edges', y | e, np.ones_like(y))):
                st[name][0] += (p & yy & care).sum((0, 2, 3)); st[name][1] += (p & ~yy & care).sum((0, 2, 3)); st[name][2] += (~p & yy & care).sum((0, 2, 3))
        for name, (tp, fp, fn) in st.items():
            pr, rc = tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)
            f1 = 2 * pr * rc / np.maximum(1e-9, pr + rc)
            print(f'{os.path.basename(pp):8s} {os.path.basename(os.path.dirname(m)) + "/" + os.path.basename(m):34s} {name:10s} F1 h {f1[0]:.3f} v {f1[1]:.3f}  prec {pr[0]:.3f} {pr[1]:.3f}  rec {rc[0]:.3f} {rc[1]:.3f}', flush=True)
