"""The rule + text-line net (RuleNet, 4 outputs: rule h, rule v, text line h, text line v) on the GPU, from text pools
(gen_pool.py: every label) and rule pools (tools/rule_net/gen_pool.py with RN_GEN=synth2: rules only, the text
outputs untrained there: weight 0).
  python train_text_v2.py OUT_DIR TEXT_POOLS RULE_POOLS VAL_TEXT VAL_RULE      (pools: comma-separated prefixes)
Environment: TN_HOURS (training time budget, default 7.5: the learning rate follows a cosine over it), TN_ITERS (an
iteration cap, default 400000), TN_CH (widths, default 12,24,32,48,64,96), TN_TEXT_P (share of text samples, 0.65),
TN_BATCH (16), TN_CROP (384), TN_SMALL_FRAC (0.75) / TN_SMALL_CROP (256) / TN_SMALL_BATCH (32): progressive resizing,
TN_COMPILE (1), TN_CL (0: channels_last), TN_LR (2e-3), TN_WORKERS (12), TN_VAL (validation every N its, 3000), TN_MEM
(GPU memory fraction, 0.85). Loss, augmentation, EMA and checkpoints as tools/rule_net/train2.py (picture edges count
as rules: RN_EDGE pos); a transposed sample swaps h and v of both pairs. OUT_DIR: best.pt (best mean F1 over the
validation pools' labelled channels), last.pt, log.jsonl. Keeps Windows awake while it runs.
v2 of train_text.py (from a GPU review, not yet run): the input in bf16 (the full-resolution cat with x was upcast to
fp32), the loss weight 4x1x1 (it has no spatial variation), uint8 targets in the workers, the loss compiled, foreach
AdamW / clipping, a lerp EMA, validation counted on the GPU, static shapes per crop size."""
import json, math, os, sys, time
import numpy as np, torch, cv2
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rule_net'))
from model import RuleNet   # noqa: E402

# the arguments (also in DataLoader workers: Windows spawns them as __mp_main__ with the same argv); none on import
A = sys.argv[1:6] if __name__ in ('__main__', '__mp_main__') else ['.', '', '', '', '']
out = A[0]
tpools, rpools = [p for p in A[1].split(',') if p], [p for p in A[2].split(',') if p]
vtext, vrule = A[3], A[4]
CH = tuple(int(c) for c in os.environ.get('TN_CH', '12,24,32,48,64,96').split(','))
B, C = int(os.environ.get('TN_BATCH', 16)), int(os.environ.get('TN_CROP', 384))
LR, HOURS, ITERS = float(os.environ.get('TN_LR', 2e-3)), float(os.environ.get('TN_HOURS', 7.5)), int(os.environ.get('TN_ITERS', 400000))
TEXT_P, VAL_EVERY = float(os.environ.get('TN_TEXT_P', 0.65)), int(os.environ.get('TN_VAL', 3000))
PERM = [1, 0, 3, 2]   # a transpose swaps horizontal and vertical
W_FULL = torch.tensor([[[1]], [[1]], [[1]], [[1]]], dtype=torch.uint8)
W_RULE = torch.tensor([[[1]], [[1]], [[0]], [[0]]], dtype=torch.uint8)


def keep_awake():
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def load(prefix):
    X = np.load(prefix + '_img.npy', mmap_mode='r')
    Y = np.load(prefix + '_mask.npy', mmap_mode='r')
    E = np.load(prefix + '_edge.npy', mmap_mode='r') if os.path.exists(prefix + '_edge.npy') else None
    return X, Y, E


def targets(y, e):
    """y uint8 2 or 4 x h x w, e edge bits or None -> target 4 x h x w, weight 4 x h x w (text weight 0 for rule pools)"""
    t = np.zeros((4,) + y.shape[1:], np.float32)
    w = np.ones_like(t)
    t[:len(y)] = y.astype(np.float32) / 255
    if len(y) == 2:
        w[2:] = 0
    if e is not None:
        for k in (0, 1):
            m = (e >> k) & 1 > 0
            t[k][m] = np.maximum(t[k][m], 1.0)
    return t, w


class Data(torch.utils.data.IterableDataset):
    def __init__(self, crop=None):
        super().__init__()
        self.crop = crop or C

    def __iter__(self):
        C = self.crop
        info = torch.utils.data.get_worker_info()
        rs = np.random.default_rng((info.id if info else 0) * 7919 + int(time.time()) % 100000)
        groups = [[load(p) for p in tpools], [load(p) for p in rpools]]
        while True:
            g = groups[0] if (rs.random() < TEXT_P or not groups[1]) else groups[1]
            X, Y, E = g[rs.integers(0, len(g))]
            i = rs.integers(0, len(X)); S = X.shape[1]
            sc = rs.uniform(0.8, 1.25) if rs.random() < 0.5 else 1.0
            cs = min(S, int(math.ceil(C / sc)))
            y0, x0 = rs.integers(0, S - cs + 1, 2)
            x = np.asarray(X[i, y0:y0 + cs, x0:x0 + cs])
            if not x.any():
                continue
            # targets stay uint8 on the labelled channels (2 or 4); picture edges set the rule channels
            tu8 = np.array(Y[i, :, y0:y0 + cs, x0:x0 + cs], copy=True)
            if E is not None:
                e_crop = np.asarray(E[i, y0:y0 + cs, x0:x0 + cs])
                tu8[0, (e_crop & 1) > 0] = 255
                tu8[1, (e_crop & 2) > 0] = 255
            if cs != C:
                x = cv2.resize(x, (C, C), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
                tu8 = cv2.resize(np.ascontiguousarray(tu8.transpose(1, 2, 0)), (C, C), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                if sc < 1:   # thinner labels after a downscale: x1.5, saturated
                    tu8 = np.minimum(tu8.astype(np.uint16) * 3 // 2, 255).astype(np.uint8)
            if len(tu8) == 2:
                t_full = np.zeros((4, C, C), dtype=np.uint8)
                t_full[:2] = tu8
                tu8, w_out = t_full, W_RULE
            else:
                w_out = W_FULL
            if rs.random() < 0.5: x = x[:, ::-1]; tu8 = tu8[:, :, ::-1]
            if rs.random() < 0.5: x = x[::-1]; tu8 = tu8[:, ::-1]
            if rs.random() < 0.5: x = x.transpose(1, 0, 2); tu8 = tu8[PERM].transpose(0, 2, 1)
            x = np.ascontiguousarray(x, dtype=np.float32) * np.float32(1.0 / 255.0)
            if rs.random() < 0.1: x = np.repeat(x.mean(2, keepdims=True), 3, 2)
            if rs.random() < 0.2: x = x * rs.uniform(0.85, 1.15, 3).astype(np.float32)
            x = (x - np.float32(0.5)) * np.float32(rs.uniform(0.7, 1.25)) + np.float32(0.5 + rs.uniform(-0.12, 0.1))
            if rs.random() < 0.15: x = cv2.GaussianBlur(x, (0, 0), rs.uniform(0.4, 1.0))
            if rs.random() < 0.2:
                x = x + rs.standard_normal(x.shape, dtype=np.float32) * np.float32(rs.uniform(0.005, 0.03))
            x_u8 = np.clip(x * np.float32(255.0) + np.float32(0.5), 0, 255).astype(np.uint8)
            if rs.random() < 0.2:
                ok, enc = cv2.imencode('.jpg', x_u8, [cv2.IMWRITE_JPEG_QUALITY, int(rs.integers(40, 90))])
                x_u8 = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            # uint8 across the worker pipe; the weight is 4x1x1 (no spatial variation)
            yield (torch.from_numpy(np.ascontiguousarray(x_u8.transpose(2, 0, 1))),
                   torch.from_numpy(np.ascontiguousarray(tu8)), w_out)


def one_thread(_):
    cv2.setNumThreads(1); torch.set_num_threads(1)


_PW = {}


def loss_fn(z, t, w):
    z = z.float()
    if z.device not in _PW:   # made once: a CUDA graph cannot capture the copy
        _PW[z.device] = torch.tensor(2.0, device=z.device)
    # the same denominator whether w is (B, 4, H, W) or (B, 4, 1, 1)
    w_scale = z.numel() // w.numel()
    bce = (F.binary_cross_entropy_with_logits(z, t, pos_weight=_PW[z.device], reduction='none') * w).sum() / (w.sum() * w_scale).clamp(min=1)
    p = torch.sigmoid(z) * w
    inter = (p * t).sum((2, 3)); den = (p + t * w).sum((2, 3))
    has = (w.sum((2, 3)) > 0).float()
    dice = ((1 - (2 * inter + 1) / (den + 1)) * has).sum() / has.sum().clamp(min=1)
    return bce + 0.5 * dice


def val_set(prefix, n_max=1500):
    X, Y, E = load(prefix)
    n = min(len(X), n_max)
    o = (X.shape[1] - 384) // 2
    Xs = np.asarray(X[:n, o:o + 384, o:o + 384])
    T = np.stack([targets(np.asarray(Y[i, :, o:o + 384, o:o + 384]), np.asarray(E[i, o:o + 384, o:o + 384]) if E is not None else None)[0]
                  for i in range(n)]) > 0.5
    return Xs, T[:, :Y.shape[1]]


def f1_counts(net, X, T):
    """per-sample tp, fp, fn per labelled channel (n, k), counted on the GPU"""
    k = T.shape[1]
    res = []
    fmt = torch.channels_last if next(net.parameters()).is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for a in range(0, len(X), 32):
            x_u8 = torch.from_numpy(X[a:a + 32]).cuda(non_blocking=True)
            x = x_u8.permute(0, 3, 1, 2).to(dtype=torch.bfloat16, memory_format=fmt).mul_(1.0 / 255.0)
            p = net(x)[:, :k] > 0
            y = torch.from_numpy(T[a:a + 32]).cuda(non_blocking=True)
            res.append(torch.stack([(p & y).sum((2, 3)), (p & ~y).sum((2, 3)), (~p & y).sum((2, 3))], -1))
    return torch.cat(res, 0).cpu().numpy()


def f1(c):
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    pr, rc = tp / np.maximum(1, tp + fp), tp / np.maximum(1, tp + fn)
    return 2 * pr * rc / np.maximum(1e-9, pr + rc)


def validate(net, VAL):
    net.eval()
    res = {}
    for name, (X, T) in VAL.items():
        c = f1_counts(net, X, T).sum(0)
        res[name] = f1(c).round(4).tolist()
    net.train()
    return res


T_START = time.time()


def train(fast):
    """fast: torch.compile and progressive resizing (else the configuration first tested: eager, full crops only)"""
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('TN_MEM', 0.85)))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    VAL = {'text': val_set(vtext), 'rules': val_set(vrule)}
    cl = fast and os.environ.get('TN_CL', '0') == '1'
    fmt = torch.channels_last if cl else torch.contiguous_format
    net = RuleNet(CH, 4).cuda().to(memory_format=fmt)
    ema = RuleNet(CH, 4).cuda().to(memory_format=fmt); ema.load_state_dict(net.state_dict()); ema.eval()
    for p in ema.parameters(): p.requires_grad_(False)
    print(f'{sum(p.numel() for p in net.parameters()) / 1e6:.3f} M params, ch {CH}, {HOURS} h, channels_last {cl}', flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4, foreach=True)
    t0 = T_START   # a fallback run continues within the same budget
    budget = HOURS * 3600

    def lr_at(it):
        frac = min(1.0, max((time.time() - t0) / budget, it / ITERS))
        return LR * min(1, (it + 1) / 500) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))

    # progressive resizing: the first TN_SMALL_FRAC of the time on TN_SMALL_CROP crops (batch TN_SMALL_BATCH, ~2.3x the
    # samples per second), then the full crop, so the net ends at its working scale
    small_frac = float(os.environ.get('TN_SMALL_FRAC', 0.75)) if fast else 0.0
    small_c, small_b = int(os.environ.get('TN_SMALL_CROP', 256)), int(os.environ.get('TN_SMALL_BATCH', 32))
    nw = int(os.environ.get('TN_WORKERS', 12))
    loader = lambda crop, b: iter(torch.utils.data.DataLoader(Data(crop), batch_size=b, num_workers=nw, pin_memory=True,
                                                              persistent_workers=False, prefetch_factor=4, worker_init_fn=one_thread))
    phase = 'small' if small_frac > 0 else 'full'
    dl = loader(small_c, small_b) if phase == 'small' else loader(C, B)
    print(f'phase {phase}', flush=True)
    # _PW made before compiling (no graph break on the dict insert inside loss_fn)
    loss_fn(torch.zeros(1, 4, 8, 8, device='cuda'), torch.zeros(1, 4, 8, 8, device='cuda'), torch.ones(1, 4, 1, 1, device='cuda'))
    fwd, loss_f = net, loss_fn
    if fast and os.environ.get('TN_COMPILE', '1') == '1':
        try:
            # static shapes: the phase change compiles a second static graph, not a slower dynamic-shape one
            fwd = torch.compile(net, dynamic=False)
            loss_f = torch.compile(loss_fn, dynamic=False)
            wb = small_b if phase == 'small' else B
            wc = small_c if phase == 'small' else C
            wx = torch.rand(wb, 3, wc, wc, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=fmt)
            wt = torch.zeros(wb, 4, wc, wc, device='cuda', dtype=torch.float32).contiguous(memory_format=fmt)
            ww = torch.ones(wb, 4, 1, 1, device='cuda', dtype=torch.float32)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                wz = fwd(wx)
            loss_f(wz, wt, ww).backward()
            opt.zero_grad(set_to_none=True)
            print('torch.compile on', flush=True)
        except Exception as e:
            fwd, loss_f = net, loss_fn
            opt.zero_grad(set_to_none=True)
            print('torch.compile failed, eager:', repr(e)[:300], flush=True)
    best, run, it = -1, 0.0, 0
    if os.path.exists(os.path.join(out, 'best.pt')):   # a fallback run keeps the first run's best unless it beats it
        best = float(np.mean([v for vs in torch.load(os.path.join(out, 'best.pt'), map_location='cpu')['val'].values() for v in vs]))
    es, ns = ema.state_dict(), net.state_dict()
    ema_f = [es[k] for k in es if es[k].dtype.is_floating_point]; net_f = [ns[k] for k in es if es[k].dtype.is_floating_point]
    ema_i = [(es[k], ns[k]) for k in es if not es[k].dtype.is_floating_point]
    logf = open(os.path.join(out, 'log.jsonl'), 'a')
    while True:
        if phase == 'small' and time.time() - t0 >= small_frac * budget:
            del dl
            phase, dl = 'full', loader(C, B)
            print(f'it {it}: phase full ({C} crops, batch {B})', flush=True)
        x, t, w = next(dl)
        for g in opt.param_groups: g['lr'] = lr_at(it)
        # x in bf16: the full-resolution cat([d, s, x]) in RuleNet.forward stays bf16
        x = x.cuda(non_blocking=True).to(dtype=torch.bfloat16, memory_format=fmt).mul_(1.0 / 255.0)
        t = t.cuda(non_blocking=True).to(dtype=torch.float32, memory_format=fmt).mul_(1.0 / 255.0)
        w = w.cuda(non_blocking=True).to(dtype=torch.float32)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = fwd(x)
        loss = loss_f(z, t, w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0, foreach=True)
        opt.step()
        d = min(0.999, (1 + it) / (10 + it))
        with torch.no_grad():   # the EMA in one foreach lerp
            torch._foreach_lerp_(ema_f, net_f, 1.0 - d)
            if it % 50 == 0:
                for pe, pn in ema_i: pe.copy_(pn)
        if it % 50 == 0:
            run = 0.9 * run + 0.1 * loss.item() if it else loss.item()
        it += 1
        if it % 500 == 0:
            print(f'{it:6d} loss {run:.4f} lr {lr_at(it):.2e} {time.time() - t0:.0f}s', flush=True)
        done = time.time() - t0 >= budget or it >= ITERS
        if it % VAL_EVERY == 0 or done:
            for pe, pn in ema_i: pe.copy_(pn)
            r = validate(ema, VAL)
            score = float(np.mean([v for vs in r.values() for v in vs]))
            print(f'  it {it} val {json.dumps(r)} mean {score:.4f}', flush=True)
            logf.write(json.dumps({'it': it, 'val': r, 'score': score, 'loss': run, 't': time.time() - t0}) + '\n'); logf.flush()
            ck = {'ch': CH, 'n_out': 4, 'state': {k: v.cpu() for k, v in ema.state_dict().items()}, 'it': it, 'val': r}
            torch.save(ck, os.path.join(out, 'last.pt'))
            if score > best:
                best = score; torch.save(ck, os.path.join(out, 'best.pt'))
        if done:
            break
    print('best mean F1', best, 'its', it, flush=True)


if __name__ == '__main__':
    os.makedirs(out, exist_ok=True)
    keep_awake()
    try:
        train(True)
    except Exception:
        import traceback
        traceback.print_exc()
        print('the compiled / progressive run failed: going on eager, full crops, within the same time budget', flush=True)
        torch.cuda.empty_cache()
        train(False)
    os._exit(0)
