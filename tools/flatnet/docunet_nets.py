"""SD-HR outputs for the DocUNet harness (docunet_bench/run_ours.py) from another SD-HR ONNX file.

  python docunet_nets.py inputs                       # pipeline_cli's SD-HR inputs (re-crop config) -> work/flatnet_in/
  python docunet_nets.py nets TAG MODEL.onnx          # -> work/nets_TAG/{n}_uv2.npz (+ uv0 / logits copied from nets_rc_f)

The inputs are the re-crop inputs of the Round 8 configuration (FS_UVPASS1 = UVDoc on the whole photo from
nets_frame_f, FS_UVRECROP=1): the crop comes from UVDoc + the page finder, never from SD-HR, so one set of inputs
serves every SD-HR checkpoint. With the shipped model, nets must reproduce nets_rc_f/{n}_uv2.npz (checked)."""
import os
import shutil
import subprocess
import sys

import numpy as np

BENCH = r'C:\Users\User\Desktop\docunet_bench'
CLI = os.environ.get('FS_CLI', r'C:\Users\User\Desktop\fs_r11\tools\desktop_check\build\pipeline_cli.exe')
WORK = os.path.join(BENCH, 'work')
NAMES = [f'{k}_{m}' for k in range(1, 66) for m in (1, 2)]
os.environ['PATH'] = r'C:\Users\User\Android\opencv\build\x64\vc16\bin;' + os.environ['PATH']


def inputs():
    od = os.path.join(WORK, 'flatnet_in')
    os.makedirs(od, exist_ok=True)
    for n in NAMES:
        out = os.path.join(od, n + '.npy')
        if os.path.exists(out):
            continue
        xin = os.path.join(od, n + '_x.npz')
        env = dict(os.environ, FS_UVPASS1=os.path.join(WORK, 'nets_frame_f', n + '_uv0.npz'), FS_UVPASS1_FRAME='1',
                   FS_UVRECROP='1', FS_UVIN=xin, FS_UVMODEL='2', FS_NO_SHADOWS='1')
        photo = os.path.join(BENCH, 'data_crop', 'crop', f'{n} copy.png')
        r = subprocess.run([CLI, photo, '-', os.path.join(WORK, 'nets_rc_f', n + '_logits.npz')], env=env,
                           capture_output=True, text=True)
        if not os.path.exists(xin):
            print(n, 'no input', r.stdout[-200:])
            continue
        x = np.load(xin)['x']
        u8 = np.round(x * 255).astype(np.uint8)
        assert np.abs(u8 / np.float32(255) - x).max() < 1e-6, n
        np.save(out, u8)
        os.remove(xin)
    print('inputs done')


def nets(tag, model):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    s = ort.InferenceSession(model, o, providers=['CPUExecutionProvider'])
    od = os.path.join(WORK, 'nets_' + tag)
    os.makedirs(od, exist_ok=True)
    src = os.path.join(WORK, 'nets_rc_f')
    for n in NAMES:
        for f in (n + '_uv0.npz', n + '_logits.npz'):
            if os.path.exists(os.path.join(src, f)) and not os.path.exists(os.path.join(od, f)):
                shutil.copy(os.path.join(src, f), os.path.join(od, f))
        ip = os.path.join(WORK, 'flatnet_in', n + '.npy')
        if not os.path.exists(ip):
            continue
        x = (np.load(ip).astype(np.float32) / np.float32(255))
        g = s.run(['grid'], {'image': x})[0]
        np.savez(os.path.join(od, n + '_uv2.npz'), out=g[0].astype(np.float32))
    print('nets', tag, 'done')


if __name__ == '__main__':
    if sys.argv[1] == 'inputs':
        inputs()
    else:
        nets(sys.argv[2], sys.argv[3])
