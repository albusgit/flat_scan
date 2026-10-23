"""The real net inputs of the 35 photos, as the app makes them (round 8 automatic path), for compare_outputs.py.

  python tools/accel/net_inputs.py BUILD_DIR OUT_DIR [PHOTO ...]
      BUILD_DIR: tools/desktop_check/build (pagefind_cli.exe, pipeline_cli.exe)
      PHOTO default: <APP_REF_DIR>/real_photos/*.jpg (34) + the hand-held receipt (500c2810-image)
  Run it with the app's ONNX Runtime (1.22.0: C:/Users/User/Android/accel_venv/Scripts/python.exe -I); the
  page finder's logits differ by ~1e-5 between ORT versions (docs/accel/NOTES.md). Several minutes for 35.

Per photo NAME, in OUT_DIR (the nets run here with onnxruntime on the CPU, the original models):
  NAME_logits.npz       page finder (pagefind_cli input -> pagefinder.onnx)
  NAME_p1_in.npz / _p1.npz    UVDoc pass 1 on the whole photo (FS_FRAME_CORNERS=1)
  NAME_rc0_in.npz / _rc0.npz  UVDoc on the re-crop around pass 1's page (FS_UVPASS1 .. FS_UVRECROP=1)
  NAME_rc2_in.npz / _rc2.npz  SyntheticDoc-HR on the same re-crop
  NAME_shade_in.npz     ShadeNet's input (planar BGR 1 x 3 x 512 x W), caught from the full pipeline run
                        with both grids (FS_UVOUT .. FS_UVOUT2, FS_SHADE_NET), as run_ours.py / the app
  NAME.json             pipeline_cli's JSON of that run
The ShadeNet input is caught by a FS_PYTHON wrapper (shade_capture.bat, written to OUT_DIR) that copies
the raw buffer pipeline_cli hands tools/shadow_net/run_onnx.py, then runs it."""
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS = os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models')
APP_REF = r'E:\fs_datasets\round4'
# the hand-held receipt (round 5 onwards; the 35th photo)
RECEIPT = (r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-flat-scan'
           r'\922e3e9d-4193-49a6-b4b4-d3421c7157d5\scratchpad\rc\photo.jpg', '500c2810-image')
os.environ['PATH'] = r'C:\Users\User\Android\opencv\build\x64\vc16\bin;' + os.environ['PATH']
_sess = {}


def net(model, x):
    import onnxruntime as ort
    if model not in _sess:
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        _sess[model] = ort.InferenceSession(os.path.join(MODELS, model), o, providers=['CPUExecutionProvider'])
    s = _sess[model]
    return s.run(None, {s.get_inputs()[0].name: x})[0][0].astype(np.float32)


def one(args):
    build, out, photo, n = args
    cli, pf = os.path.join(build, 'pipeline_cli.exe'), os.path.join(build, 'pagefind_cli.exe')
    p = lambda s: os.path.join(out, n + s)
    if not os.path.exists(p('_logits.npz')):
        subprocess.run([pf, 'input', photo, p('_pfx.npz')], check=True, capture_output=True)
        np.savez(p('_logits.npz'), logits=net('pagefinder.onnx', np.load(p('_pfx.npz'))['x']))
        os.remove(p('_pfx.npz'))
    pass1 = dict(FS_UVPASS1=p('_p1.npz'), FS_UVPASS1_FRAME='1', FS_UVRECROP='1')
    for tag, model, onnx, extra in (('p1', 0, 'uvdoc.onnx', dict(FS_FRAME_CORNERS='1')),
                                    ('rc0', 0, 'uvdoc.onnx', pass1), ('rc2', 2, 'syntheticdoc_hr.onnx', pass1)):
        if os.path.exists(p(f'_{tag}.npz')):
            continue
        env = dict(os.environ, FS_UVIN=p(f'_{tag}_in.npz'), FS_UVMODEL=str(model), **extra)
        r = subprocess.run([cli, photo, '-', p('_logits.npz')], env=env, capture_output=True, text=True)
        if not os.path.exists(p(f'_{tag}_in.npz')):
            return n, f'no {tag} input: {r.stdout[-200:]}'
        np.savez(p(f'_{tag}.npz'), out=net(onnx, np.load(p(f'_{tag}_in.npz'))['x']))
    if not os.path.exists(p('_shade_in.npz')):
        env = dict(os.environ, FS_UVOUT=p('_rc2.npz'), FS_UVMODEL='2', FS_UVOUT2=p('_rc0.npz'), FS_UVMODEL2='0',
                   FS_SHADE_NET=os.path.join(MODELS, 'shadenet.onnx'), FS_PYTHON=os.path.join(out, 'shade_capture.bat'),
                   **pass1)
        r = subprocess.run([cli, photo, p('.jpg'), p('_logits.npz')], env=env, capture_output=True, text=True)
        open(p('.json'), 'w').write(r.stdout)
        cap = p('.jpg.shin.f32.cap')
        if not os.path.exists(cap):
            return n, 'no shade net input: ' + r.stdout[-200:]
        w, h = (int(v) for v in open(cap + '.txt').read().split())
        np.savez(p('_shade_in.npz'), x=np.fromfile(cap, np.float32).reshape(1, 3, h, w))
        for f in (cap, cap + '.txt', p('.jpg.shout.f32')):
            if os.path.exists(f):
                os.remove(f)
    return n, 'ok'


def main():
    build, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    py = os.path.join(os.path.dirname(sys.executable), 'python.exe')
    with open(os.path.join(out, 'shade_capture.bat'), 'w') as f:
        # called as: shade_capture.bat -I run_onnx.py MODEL IN.f32 OUT.f32 W H
        f.write('@echo off\r\ncopy /y "%~4" "%~4.cap" >nul\r\necho %6 %7> "%~4.cap.txt"\r\n'
                f'"{py}" %*\r\n')
    photos = sys.argv[3:] or sorted(glob.glob(os.path.join(APP_REF, 'real_photos', '*.jpg'))) + [RECEIPT[0]]
    jobs = [(build, out, ph, RECEIPT[1] if ph == RECEIPT[0] else os.path.splitext(os.path.basename(ph))[0]) for ph in photos]
    with Pool(4) as pool:
        for n, msg in pool.imap_unordered(one, jobs):
            print(f'{n:36s} {msg}', flush=True)
    print(len(jobs), 'photos')


if __name__ == '__main__':
    main()
