"""Train RuleNet on pools from gen_pool.py (RN_CROP, RN_BATCH: crop size and batch; RN_LR: peak learning rate).
  python train.py POOL VALPOOL OUT_DIR [ITERS] [RESUME.pt]
Batches of 384 px crops; flips and transposes (a transposed page swaps the two masks: stage-1
pages can be sideways, their orientation is decided later), brightness / contrast / grey jitter.
Loss: BCE (positives weighted 2x: rules are a few % of pixels) + soft Dice per mask. Prints the
validation F1 of each mask at 0.5 and saves OUT_DIR/rulenet.pt (best F1) and last.pt."""
import math, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RuleNet

pool, vpool, out = sys.argv[1:4]
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
resume = sys.argv[5] if len(sys.argv) > 5 else None
os.makedirs(out, exist_ok=True)
torch.set_num_threads(os.cpu_count())
torch.manual_seed(0)
rs = np.random.default_rng(0)
X = np.load(pool + '_img.npy', mmap_mode='r'); Y = np.load(pool + '_mask.npy', mmap_mode='r')
VX = np.load(vpool + '_img.npy')[:, 8:392, 8:392]; VY = np.load(vpool + '_mask.npy')[:, :, 8:392, 8:392]
N, S = X.shape[0], X.shape[1]
C, B = int(os.environ.get("RN_CROP", 384)), int(os.environ.get("RN_BATCH", 12))


def batch():
    idx = rs.integers(0, N, B)
    xs, ys = [], []
    for i in idx:
        y0, x0 = rs.integers(0, S - C + 1, 2)
        x = np.asarray(X[i, y0:y0 + C, x0:x0 + C], np.float32) / 255
        y = np.asarray(Y[i, :, y0:y0 + C, x0:x0 + C], np.float32) / 255
        if rs.random() < 0.5: x = x[:, ::-1]; y = y[:, :, ::-1]
        if rs.random() < 0.5: x = x[::-1]; y = y[:, ::-1]
        if rs.random() < 0.5: x = x.transpose(1, 0, 2); y = y[::-1].transpose(0, 2, 1)
        if rs.random() < 0.1: x = np.repeat(x.mean(2, keepdims=True), 3, 2)
        x = np.clip((x - 0.5) * rs.uniform(0.75, 1.25) + 0.5 + rs.uniform(-0.1, 0.1), 0, 1)
        xs.append(x.transpose(2, 0, 1)); ys.append(y)
    return torch.from_numpy(np.ascontiguousarray(np.stack(xs))), torch.from_numpy(np.ascontiguousarray(np.stack(ys)))


def loss_fn(z, y):
    bce = F.binary_cross_entropy_with_logits(z, y, pos_weight=torch.tensor(2.0))
    p = torch.sigmoid(z)
    inter = (p * y).sum((2, 3)); den = (p + y).sum((2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return bce + 0.5 * dice


def validate(net):
    net.eval()
    tp = np.zeros(2); fp = np.zeros(2); fn = np.zeros(2)
    with torch.no_grad():
        for k in range(0, len(VX), 16):
            x = torch.from_numpy(VX[k:k + 16].astype(np.float32).transpose(0, 3, 1, 2) / 255)
            p = (torch.sigmoid(net(x)) > 0.5).numpy()
            y = VY[k:k + 16] > 127
            tp += (p & y).sum((0, 2, 3)); fp += (p & ~y).sum((0, 2, 3)); fn += (~p & y).sum((0, 2, 3))
    net.train()
    prec, rec = tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)
    return 2 * prec * rec / np.maximum(1e-9, prec + rec), prec, rec


net = RuleNet()
if resume:
    net.load_state_dict(torch.load(resume, map_location='cpu'))
opt = torch.optim.AdamW(net.parameters(), lr=float(os.environ.get("RN_LR", 3e-3)), weight_decay=1e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 200) * 0.5 * (1 + math.cos(math.pi * i / iters)))
best, t0, run = -1, time.time(), 0.0
for it in range(iters):
    x, y = batch()
    loss = loss_fn(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
    if it % 100 == 0:
        print(f'{it:6d} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s', flush=True)
    if (it + 1) % 1000 == 0 or it + 1 == iters:
        f1, pr, rc = validate(net)
        print(f'  val F1 h {f1[0]:.3f} v {f1[1]:.3f}  prec {pr.round(3)} rec {rc.round(3)}', flush=True)
        torch.save(net.state_dict(), os.path.join(out, 'last.pt'))
        if f1.mean() > best:
            best = f1.mean(); torch.save(net.state_dict(), os.path.join(out, 'rulenet.pt'))
print('best mean F1', best)
