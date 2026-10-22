"""Apply ShadeNet to a set's inputs as the app does (pipeline.cpp apply_shade_net: page at long side 512, portrait,
map clamped to [0.3, 1], page divided by it): python -I shade_set.py MODEL.onnx SRCSET DSTSET"""
import glob, os, shutil, sys
import numpy as np, cv2, onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider'])
src, dst = sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
iname = sess.get_inputs()[0].name
for f in sorted(glob.glob(os.path.join(src, '*_in.png'))):
    bgr = cv2.imread(f)
    turned = bgr.shape[1] > bgr.shape[0]
    b = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE) if turned else bgr
    s = 512 / max(b.shape[:2])
    nw, nh = max(32, round(b.shape[1] * s / 32) * 32), max(32, round(b.shape[0] * s / 32) * 32)
    x = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    S = sess.run(None, {iname: x.transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
    S = np.clip(S, 0.3, 1.0)
    if turned:
        S = cv2.rotate(S, cv2.ROTATE_90_COUNTERCLOCKWISE)
    S = cv2.resize(S, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = np.clip(bgr / S + 0.5, 0, 255).astype(np.uint8)
    n = os.path.basename(f)[:-7]
    cv2.imwrite(os.path.join(dst, n + '_in.png'), out)
    shutil.copy(f.replace('_in.png', '_gt.png'), os.path.join(dst, n + '_gt.png'))
print('done')
