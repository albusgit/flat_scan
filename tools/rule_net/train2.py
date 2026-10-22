"""Round 11: train RuleNet on the GPU from pools of gen_pool.py (synth.py and synth2.py pools).
  python train2.py OUT_DIR TRAIN_POOLS VAL_POOLS ITERS     (pools: comma-separated prefixes)
Environment: RN_CH (model widths, default 8,16,24,32,48,64 = round 10), RN_BATCH (16), RN_CROP (384),
RN_LR (2e-3), RN_EDGE (pos | ignore | neg: what the straight edges of pictures / dark bars recorded in
OUT_edge.npy count as, default pos), RN_MEM (GPU memory fraction, default 0.09 = 0.7 GB of 8),
RN_WORKERS (4), RN_INIT (a checkpoint to start from).

Augmentation (in workers): crops, flips and transposes (a transposed page swaps the masks), a scale
of 0.8 - 1.25, brightness / contrast / grey / colour jitter, blur, noise and a JPEG pass.
Loss: BCE (positives x2) + 0.5 soft Dice. bf16 autocast, AdamW, warm-up and cosine decay, an EMA of
the weights. Every RN_VAL iterations the EMA model is scored on each validation pool (F1 of each mask
at 0.5, 384-px centre crops); OUT_DIR/best.pt keeps the best mean F1, last.pt the latest
({'ch': widths, 'state': state_dict}); log.jsonl the history."""
import json, math, os, sys, time
import numpy as np, torch, cv2
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RuleNet

out, tpools, vpools, iters = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(','), int(sys.argv[4])
CH = tuple(int(c) for c in os.environ.get('RN_CH', '8,16,24,32,48,64').split(','))
B, C = int(os.environ.get('RN_BATCH', 16)), int(os.environ.get('RN_CROP', 384))
LR, EDGE = float(os.environ.get('RN_LR', 2e-3)), os.environ.get('RN_EDGE', 'pos')
VAL_EVERY = int(os.environ.get('RN_VAL', 2000))
os.makedirs(out, exist_ok=True)


def load(prefix, mmap=True):
    X = np.load(prefix + '_img.npy', mmap_mode='r' if mmap else None)
    Y = np.load(prefix + '_mask.npy', mmap_mode='r' if mmap else None)
    E = np.load(prefix + '_edge.npy', mmap_mode='r' if mmap else None) if os.path.exists(prefix + '_edge.npy') else None
    return X, Y, E


def targets(y, e):
    """y uint8 2 x h x w, e uint8 h x w bits (or None) -> target float 2 x h x w, weight 2 x h x w"""
    t = y.astype(np.float32) / 255
    w = np.ones_like(t)
    if e is not None and EDGE != 'neg':
        for k in (0, 1):
            m = (e >> k) & 1 > 0
            if EDGE == 'pos':
                t[k][m] = np.maximum(t[k][m], 1.0)
            else:
                w[k][m & (t[k] < 0.5)] = 0
    return t, w


class Data(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rs = np.random.default_rng((info.id if info else 0) * 7919 + int(time.time()) % 100000)
        pools, sizes, n = [], None, 0
        while True:
            if n % 2000 == 0:   # pools still being generated join once their log says done
                ready = [p for p in tpools if os.path.exists(p + '_img.npy') and (not os.path.exists(p + '.log') or 'done' in open(p + '.log').read())]
                if len(ready) != len(pools):
                    pools = [load(p) for p in ready]
                    sizes = np.array([p[0].shape[0] for p in pools], np.float64)
            n += 1
            pi = rs.choice(len(pools), p=sizes / sizes.sum())
            X, Y, E = pools[pi]
            i = rs.integers(0, X.shape[0]); S = X.shape[1]
            sc = rs.uniform(0.8, 1.25) if rs.random() < 0.5 else 1.0
            cs = min(S, int(math.ceil(C / sc)))
            y0, x0 = rs.integers(0, S - cs + 1, 2)
            x = np.asarray(X[i, y0:y0 + cs, x0:x0 + cs])
            y = np.asarray(Y[i, :, y0:y0 + cs, x0:x0 + cs])
            e = np.asarray(E[i, y0:y0 + cs, x0:x0 + cs]) if E is not None else None
            t, w = targets(y, e)
            if cs != C:
                x = cv2.resize(x, (C, C), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
                t = np.stack([cv2.resize(q, (C, C), interpolation=cv2.INTER_LINEAR) for q in t])
                w = np.stack([cv2.resize(q, (C, C), interpolation=cv2.INTER_NEAREST) for q in w])
                if sc < 1:   # thinner than a pixel: a rule pixel is one the rule covers at all
                    t = np.clip(t * 1.5, 0, 1)
            x = x.astype(np.float32) / 255
            if rs.random() < 0.5: x = x[:, ::-1]; t = t[:, :, ::-1]; w = w[:, :, ::-1]
            if rs.random() < 0.5: x = x[::-1]; t = t[:, ::-1]; w = w[:, ::-1]
            if rs.random() < 0.5: x = x.transpose(1, 0, 2); t = t[::-1].transpose(0, 2, 1); w = w[::-1].transpose(0, 2, 1)
            x = np.ascontiguousarray(x)
            if rs.random() < 0.1: x = np.repeat(x.mean(2, keepdims=True), 3, 2)
            if rs.random() < 0.2: x = x * rs.uniform(0.85, 1.15, 3).astype(np.float32)
            x = (x - 0.5) * rs.uniform(0.7, 1.25) + 0.5 + rs.uniform(-0.12, 0.1)
            if rs.random() < 0.15: x = cv2.GaussianBlur(x, (0, 0), rs.uniform(0.4, 1.0))
            if rs.random() < 0.2: x = x + rs.normal(0, rs.uniform(0.005, 0.03), x.shape).astype(np.float32)
            x = np.clip(x, 0, 1)
            if rs.random() < 0.2:
                ok, enc = cv2.imencode('.jpg', (x * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rs.integers(40, 90))])
                x = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255
            yield (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(t)),
                   torch.from_numpy(np.ascontiguousarray(w)))


def loss_fn(z, t, w):
    z = z.float()
    bce = (F.binary_cross_entropy_with_logits(z, t, pos_weight=torch.tensor(2.0, device=z.device), reduction='none') * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(z) * w
    inter = (p * t).sum((2, 3)); den = (p + t * w).sum((2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return bce + 0.5 * dice


VAL = []
for vp in vpools:
    X, Y, E = load(vp, mmap=False)
    o = (X.shape[1] - 384) // 2
    X = X[:, o:o + 384, o:o + 384]; Y = Y[:, :, o:o + 384, o:o + 384]
    E = E[:, o:o + 384, o:o + 384] if E is not None else None
    T = np.stack([targets(Y[i], E[i] if E is not None else None)[0] for i in range(len(X))]) > 0.5
    VAL.append((os.path.basename(vp), X, T))


def validate(net):
    net.eval()
    res = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for name, X, T in VAL:
            tp = np.zeros(2); fp = np.zeros(2); fn = np.zeros(2)
            for k in range(0, len(X), 8):
                x = torch.from_numpy(X[k:k + 8].astype(np.float32).transpose(0, 3, 1, 2) / 255).cuda()
                p = (net(x).float() > 0).cpu().numpy()
                y = T[k:k + 8]
                tp += (p & y).sum((0, 2, 3)); fp += (p & ~y).sum((0, 2, 3)); fn += (~p & y).sum((0, 2, 3))
            prec, rec = tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)
            f1 = 2 * prec * rec / np.maximum(1e-9, prec + rec)
            res[name] = {'f1': f1.round(4).tolist(), 'prec': prec.round(4).tolist(), 'rec': rec.round(4).tolist()}
    net.train()
    return res


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('RN_MEM', 0.09)))
    torch.backends.cudnn.benchmark = True
    net = RuleNet(CH).cuda()
    if os.environ.get('RN_INIT'):
        ck = torch.load(os.environ['RN_INIT'], map_location='cpu')
        net.load_state_dict(ck['state'] if 'state' in ck else ck)
    ema = RuleNet(CH).cuda(); ema.load_state_dict(net.state_dict()); ema.eval()
    for p in ema.parameters(): p.requires_grad_(False)
    print(f'{sum(p.numel() for p in net.parameters()) / 1e6:.3f} M params, ch {CH}, edges {EDGE}', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1, i / iters)))))
    dl = torch.utils.data.DataLoader(Data(), batch_size=B, num_workers=int(os.environ.get('RN_WORKERS', 4)), pin_memory=True,
                                     persistent_workers=True, prefetch_factor=4)
    best, t0, run = -1, time.time(), 0.0
    logf = open(os.path.join(out, 'log.jsonl'), 'a')
    it = 0
    for x, t, w in dl:
        x, t, w = x.cuda(non_blocking=True), t.cuda(non_blocking=True), w.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = net(x)
        loss = loss_fn(z, t, w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step(); sched.step()
        d = min(0.999, (1 + it) / (10 + it))
        with torch.no_grad():
            for pe, pn in zip(ema.state_dict().values(), net.state_dict().values()):
                if pe.dtype.is_floating_point: pe.mul_(d).add_(pn.detach(), alpha=1 - d)
                else: pe.copy_(pn)
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if it % 200 == 0:
            print(f'{it:6d} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MB', flush=True)
        it += 1
        if it % VAL_EVERY == 0 or it == iters:
            r = validate(ema)
            score = float(np.mean([np.mean(v['f1']) for v in r.values()]))
            print(f'  it {it} val', json.dumps(r), f'mean {score:.4f}', flush=True)
            logf.write(json.dumps({'it': it, 'val': r, 'score': score, 'loss': run, 't': time.time() - t0}) + '\n'); logf.flush()
            ck = {'ch': CH, 'state': {k: v.cpu() for k, v in ema.state_dict().items()}, 'it': it, 'val': r}
            torch.save(ck, os.path.join(out, 'last.pt'))
            if it % 10000 == 0:
                torch.save(ck, os.path.join(out, f'it{it}.pt'))
            if score > best:
                best = score; torch.save(ck, os.path.join(out, 'best.pt'))
        if it >= iters:
            break
    print('best mean F1', best, flush=True)
    os._exit(0)
