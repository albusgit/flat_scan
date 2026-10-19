"""The app's pipeline with UVDoc (core/src/uvdoc.cpp via tools/desktop_check pipeline_cli) on photos.
  python tools/uvdoc_check.py UVDOC.onnx PIPELINE_CLI LOGITS_DIR OUTDIR PHOTO...
  (FS_UVMODEL=1: SyntheticDoc, whose second output, the shading map, is passed on too)
LOGITS_DIR holds <photo>_logits.npz (page-finder output, tools/pagefind_parity.py). Per photo:
pipeline_cli writes the net input for the detected corners (FS_UVIN), the net runs here
(onnxruntime, CPU, as the app runs it), pipeline_cli processes with its output (FS_UVOUT).
Writes OUTDIR/<photo>.jpg and prints the pipeline's JSON (mode, uv_agreement, uv_rejected)."""
import json, os, subprocess, sys
import numpy as np, onnxruntime as ort

model, cli, logits_dir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
for photo in sys.argv[5:]:
    n = os.path.splitext(os.path.basename(photo))[0]
    lg = os.path.join(logits_dir, n + '_logits.npz')
    env = dict(os.environ, FS_UVIN=os.path.join(out, n + '_uvin.npz'))
    subprocess.run([cli, photo, '-', lg], env=env, check=True, capture_output=True)
    x = np.load(env['FS_UVIN'])['x']
    ys = sess.run(None, {sess.get_inputs()[0].name: x})
    arrays = {'out': ys[0][0].astype(np.float32)}
    if len(ys) > 1:
        arrays['shading'] = ys[1][0].astype(np.float32)
    np.savez(os.path.join(out, n + '_uvout.npz'), **arrays)
    env = dict(os.environ, FS_UVOUT=os.path.join(out, n + '_uvout.npz'))
    r = subprocess.run([cli, photo, os.path.join(out, n + '.jpg'), lg], env=env, capture_output=True, text=True)
    j = json.loads(r.stdout)
    print(f"{n:36s} mode {j['mode']:10s} uv agreement {j.get('uv_agreement')} rejected {j.get('uv_rejected')}", flush=True)
