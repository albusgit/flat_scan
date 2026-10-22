"""compare_rule.py RULENET_ONNX RULENET_TFLITE PAGE [PAGE ...]: the rule net on LiteRT (CPU, fp32, the default
XNNPACK delegate, input resized to each page: the app's loader does the same) against ONNX Runtime CPU, on the
inputs core/src/warpmap.cpp rule_probs makes: the page at half size (INTER_AREA), replicate-padded to multiples
of 32, BGR / 255, planar. Prints per page max / p99 of |probability difference| and the share of pixels whose
> 0.5 decision flips, then the pooled numbers. Run with C:/Users/User/Android/litert_venv (onnxruntime 1.22.0,
ai-edge-litert); needs cv2 there or --npz inputs made elsewhere."""
import sys
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter


def rule_input(bgr):
    import cv2
    h, w = bgr.shape[:2]
    w2, h2 = (w + 1) // 2, (h + 1) // 2
    pw, ph = (w2 + 31) // 32 * 32, (h2 + 31) // 32 * 32
    small = cv2.resize(bgr, (w2, h2), interpolation=cv2.INTER_AREA)
    pad = cv2.copyMakeBorder(small, 0, ph - h2, 0, pw - w2, cv2.BORDER_REPLICATE)
    return pad.astype(np.float32).transpose(2, 0, 1)[None] / np.float32(255)


def main():
    onnx_path, tfl_path, pages = sys.argv[1], sys.argv[2], sys.argv[3:]
    s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    it = Interpreter(model_path=tfl_path, num_threads=4)
    inp = it.get_input_details()[0]['index']
    diffs, flips, n = [], 0, 0
    if len(pages) == 1 and pages[0].endswith('.npz'):
        z = np.load(pages[0])
        items = [(k, z[k]) for k in z.files]
    else:
        import cv2
        items = [(p, rule_input(cv2.imread(p))) for p in pages]
    for name, x in items:
        a = s.run(None, {s.get_inputs()[0].name: x})[0]
        it.resize_tensor_input(inp, [1, x.shape[2], x.shape[3], 3], strict=True)
        it.allocate_tensors()
        it.set_tensor(inp, np.ascontiguousarray(x.transpose(0, 2, 3, 1)))
        it.invoke()
        b = it.get_tensor(it.get_output_details()[0]['index']).transpose(0, 3, 1, 2)
        d = np.abs(a.astype(np.float64) - b)
        f = int(np.count_nonzero((a > 0.5) != (b > 0.5)))
        diffs.append(d.ravel())
        flips += f
        n += a.size
        print(f'{str(name)[-40:]:40s} {x.shape[2]}x{x.shape[3]}  max {d.max():.6f}  p99 {np.percentile(d, 99):.6f}  flips {f}', flush=True)
    d = np.concatenate(diffs)
    print(f'pooled over {len(items)} pages: max {d.max():.6f}  p99 {np.percentile(d, 99):.6f}  flips {flips} of {n} ({100 * flips / n:.5f} %)')


if __name__ == '__main__':
    main()
