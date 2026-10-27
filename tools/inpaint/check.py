"""MI-GAN TFLite vs PyTorch on the test cases of tools/inpaint/export_onnx.py, side-by-side PNGs, CPU timing.

  python -I tools/inpaint/check.py CASES_NPZ TFLITE_DIR OUT_DIR [--runs N] [--threads T,T,...]
      CASES_NPZ: cases.npz of export_onnx.py (NHWC net inputs, PyTorch and ONNX Runtime outputs)
      TFLITE_DIR: migan_512.tflite (fp16 weights) and migan_512_fp32.tflite of tools/inpaint/convert.py
      -> per case and file: max / mean |TFLite - PyTorch| on the raw output (whole 512 x 512, and in the hole only),
         and the same in 8-bit pixels after the post-processing ((y * 0.5 + 0.5) * 255, rounded, clamped) in the hole
      -> OUT_DIR/NAME.png: input with the hole tinted | PyTorch result | TFLite (fp16 weights) result |
         |TFLite - PyTorch| x 20 (results: the net's output in the hole, the input elsewhere, as an app composes it)
      -> CPU time of tf.lite.Interpreter.invoke() (XNNPACK, the interpreter's default CPU path) for each thread count
         (default 1 and 4): median of N runs (default 10) after one warm-up
Run it with C:/Users/User/Android/litert_venv (tensorflow 2.19.1). PNGs are written with zlib (no PIL in the venv)."""
import os
import statistics
import struct
import sys
import time
import zlib

import numpy as np


def write_png(path, rgb):
    """rgb uint8 [H, W, 3] -> 8-bit RGB PNG."""
    h, w, _ = rgb.shape
    raw = b''.join(b'\0' + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def to_u8(y):
    """Net output (RGB, ~[-1, 1]) -> uint8 pixels: (y * 0.5 + 0.5) * 255, rounded, clamped."""
    return np.clip(np.rint((y * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def main():
    import tensorflow as tf
    args = sys.argv[1:]
    runs, threads = 10, [1, 4]
    if '--runs' in args:
        k = args.index('--runs')
        runs = int(args[k + 1])
        del args[k:k + 2]
    if '--threads' in args:
        k = args.index('--threads')
        threads = [int(t) for t in args[k + 1].split(',')]
        del args[k:k + 2]
    cases, tdir, out = np.load(args[0]), args[1], os.path.abspath(args[2])
    os.makedirs(out, exist_ok=True)
    files = [('fp16w', os.path.join(tdir, 'migan_512.tflite')), ('fp32', os.path.join(tdir, 'migan_512_fp32.tflite'))]
    print(f'tensorflow {tf.__version__}; ' + ', '.join(f'{os.path.basename(p)} {os.path.getsize(p)} bytes'
                                                      for _, p in files))
    its = {}
    for tag, p in files:
        it = tf.lite.Interpreter(model_path=p)
        it.allocate_tensors()
        its[tag] = it
    for name in cases['names']:
        name = str(name)
        x, yt = cases[name + '/x_nhwc'], cases[name + '/y_torch_nhwc']
        rgb, hole = cases[name + '/rgb'], cases[name + '/hole'] > 0
        res = {}
        for tag, it in its.items():
            it.set_tensor(it.get_input_details()[0]['index'], x)
            it.invoke()
            y = it.get_tensor(it.get_output_details()[0]['index']).copy()
            res[tag] = y
            d = np.abs(y - yt)[0]
            d8 = np.abs(to_u8(y[0]).astype(int) - to_u8(yt[0]).astype(int))[hole]
            print(f'{name} {tag}: |tflite - torch| raw max {d.max():.2e} mean {d.mean():.2e}; in the hole max '
                  f'{d[hole].max():.2e} mean {d[hole].mean():.2e}; 8-bit in the hole max {d8.max()} mean {d8.mean():.3f}, '
                  f'{100 * (d8 > 1).mean():.2f}% of hole pixel channels off by > 1', flush=True)
        do = np.abs(cases[name + '/y_onnx_nhwc'] - yt)[0]
        print(f'{name} onnxruntime (export): |onnx - torch| raw max {do.max():.2e} mean {do.mean():.2e}')
        tint = rgb.copy()
        tint[hole] = (0.45 * rgb[hole] + 0.55 * np.array([255, 0, 64])).astype(np.uint8)
        hm = hole[..., None]
        comp_t = np.where(hm, to_u8(yt[0]), rgb)
        comp_l = np.where(hm, to_u8(res['fp16w'][0]), rgb)
        diff = np.clip(np.abs(res['fp16w'][0] - yt[0]) * 0.5 * 255 * 20, 0, 255).astype(np.uint8)
        sep = np.full((rgb.shape[0], 8, 3), 255, np.uint8)
        write_png(os.path.join(out, name + '.png'), np.hstack([tint, sep, comp_t, sep, comp_l, sep, diff]))
    x = cases[str(cases['names'][0]) + '/x_nhwc']
    for tag, p in files:
        for t in threads:
            it = tf.lite.Interpreter(model_path=p, num_threads=t)
            it.allocate_tensors()
            it.set_tensor(it.get_input_details()[0]['index'], x)
            it.invoke()
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter()
                it.invoke()
                ts.append(time.perf_counter() - t0)
            print(f'{os.path.basename(p)} CPU {t} thread(s): median {1000 * statistics.median(ts):.0f} ms, min '
                  f'{1000 * min(ts):.0f} ms over {runs} runs', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
