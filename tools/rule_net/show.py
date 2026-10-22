"""The rule net's masks on pages: python show.py CKPT.pt|MODEL.onnx OUT.jpg PAGE...
Each page (a stage-1 page, any size) at half size through the net; red: horizontal rule > 0.5,
blue: vertical rule > 0.5, side by side with the page."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

model, out = sys.argv[1], sys.argv[2]
if model.endswith('.onnx'):
    import onnxruntime as ort
    s = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    run = lambda x: s.run(None, {'page': x})[0]
else:
    import torch
    from model import RuleNet
    net = RuleNet().eval(); net.load_state_dict(torch.load(model, map_location='cpu'))
    def run(x):
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(x))).numpy()
rows = []
for f in sys.argv[3:]:
    im = cv2.imread(f)
    h2, w2 = (im.shape[0] + 1) // 2, (im.shape[1] + 1) // 2
    sm = cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA)
    ph, pw = (h2 + 31) // 32 * 32, (w2 + 31) // 32 * 32
    pad = cv2.copyMakeBorder(sm, 0, ph - h2, 0, pw - w2, cv2.BORDER_REPLICATE)
    p = run(pad.astype(np.float32).transpose(2, 0, 1)[None] / 255)[0, :, :h2, :w2]
    v = sm.copy()
    v[p[0] > 0.5] = (0, 0, 255); v[p[1] > 0.5] = (255, 0, 0)
    r = np.hstack([sm, v])
    rows.append(cv2.resize(r, (1200, int(1200 * r.shape[0] / r.shape[1]))))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
