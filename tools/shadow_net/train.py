"""Train ShadeNet on synth.py samples.
  python train.py RUN_DIR [STEPS] [INIT.pt]
Loss: L1 of log shading (= relative error of the de-shadowed page) + 0.5 x L1 of its gradient
(sharp shadow edges, no text in the map). Every 1000 steps: loss on a fixed synthetic validation set
and a sheet of real flattened pages (input | map | divided) in RUN_DIR."""
import glob, os, random, sys, time
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')   # per-thread BLAS buffers cost ~0.5 GB per worker
import numpy as np, cv2
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth
if __name__ == '__main__':   # not in the spawned data workers
    import torch
    from model import ShadeNet

H, W = 512, 384
BATCH = 16
REAL = r'C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Desktop-flat-scan\922e3e9d-4193-49a6-b4b4-d3421c7157d5\scratchpad\uvshare'
REAL_NAMES = ['docres__208_in', 'docres__for_deshadowing', 'docscanner__42_2_copy', '500c2810-image', 'user__projector_screen',
              'docres__for_appearance', 'docres__190_in', 'scanner_samples__chart', 'docres__151_in', 'doctr__25']


def gen(seed):
    """half a batch of pages and of half-size shading maps from a plain worker process (numpy /
    OpenCV only: torch workers cost ~0.5 GB each); the GPU pairs them up twice (compose). None when
    the worker fails (out of memory while other jobs run): skipped, the pool keeps going."""
    try:
        return _gen(seed)
    except Exception:
        return None


def _gen(seed):
    rng = random.Random(seed)
    A, T = [], []
    for _ in range(BATCH // 2):
        A.append((synth.page_albedo(rng, H, W) * 255 + 0.5).astype(np.uint8))
        L, _ = synth.shading(rng, H // 2, W // 2)
        ref = np.percentile(cv2.resize(L, (W // 16, H // 16), interpolation=cv2.INTER_AREA).reshape(-1, 3), 99.5, axis=0)
        T.append(np.clip(L / ref, 0.02, 1.0).astype(np.float16))
    return np.stack(A), np.stack(T)


def compose(A, T, dev):
    """synth.sample on the GPU (no JPEG: the app's net input is a downscaled page): albedo x
    shading x colour cast x exposure, camera blur, shot noise, 8-bit levels. Each page and each
    shading map is used twice (two pairings)."""
    import torchvision.transforms.functional as TF
    a = A.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    t = T.to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
    t = torch.nn.functional.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
    n = a.shape[0]
    perm = torch.randperm(n, device=dev)
    a = torch.cat([a, a]); t = torch.cat([t, t[perm]])
    B = a.shape[0]
    cast = torch.empty(B, 3, 1, 1, device=dev).uniform_(0.9, 1.04)
    expo = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.75, 1.15)
    o = a * t * cast * expo
    for i in range(B):
        if random.random() < 0.6:
            sg = random.uniform(0.3, 1.1)
            o[i] = TF.gaussian_blur(o[i:i + 1], kernel_size=2 * int(3 * sg) + 1, sigma=sg)[0]
    o = o.clamp_(0, 1)
    sig = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.002, 0.02) * 1.5 * (torch.rand(B, 1, 1, 1, device=dev) < 0.8)
    o = (o + torch.randn_like(o) * sig * o.clamp(min=0.02).sqrt()).clamp_(0, 1)
    o = torch.round(o * 255) / 255
    return o.contiguous(memory_format=torch.channels_last), t


def batches(workers):
    from multiprocessing import Pool
    base = int(time.time()) % 100000 * 1000
    pool = Pool(workers)
    for r in pool.imap(gen, (base + i for i in range(10 ** 9)), chunksize=1):
        if r is None:
            continue
        A, T = r
        yield torch.from_numpy(A).pin_memory(), torch.from_numpy(T).pin_memory()


def grad(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def loss_fn(p, t):
    lp, lt = torch.log(p + 0.02), torch.log(t + 0.02)
    l1 = (lp - lt).abs().mean()
    gy, gx = grad(lp); ty, tx = grad(lt)
    lg = (gy - ty).abs().mean() + (gx - tx).abs().mean()
    return l1 + 0.5 * lg, l1


def fit_page(img, long_side=512):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def real_sheet(net, dev, path):
    tiles = []
    for n in REAL_NAMES:
        p = os.path.join(REAL, n + '.jpg')
        if not os.path.exists(p):
            continue
        im = fit_page(cv2.imread(p))
        x = torch.from_numpy((im.astype(np.float32) / 255).transpose(2, 0, 1)[None]).to(dev)
        with torch.no_grad():
            S = net(x)[0].float().cpu().numpy().transpose(1, 2, 0)
        S = np.minimum(S, 1)
        div = np.clip(im.astype(np.float32) / 255 / np.maximum(S, 0.05), 0, 1)
        t = np.hstack([im, (S * 255).astype(np.uint8), (div * 255).astype(np.uint8)])
        t = cv2.resize(t, (int(t.shape[1] * 360 / t.shape[0]), 360))
        tiles.append(t)
    rows, cur = [], []
    for t in tiles:
        cur.append(t)
        if len(cur) == 2:
            rows.append(cur); cur = []
    if cur:
        rows.append(cur)
    Wd = max(sum(x.shape[1] for x in r) for r in rows)
    out = np.vstack([np.hstack(r + [np.zeros((360, Wd - sum(x.shape[1] for x in r), 3), np.uint8)]) for r in rows])
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    run = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    os.makedirs(run, exist_ok=True)
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    net = ShadeNet()
    if len(sys.argv) > 3:
        net.load_state_dict(torch.load(sys.argv[3], map_location='cpu'))
    net = net.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    # fixed validation set
    vr = random.Random(12345)
    val = [synth.sample(vr, H, W)[:2] for _ in range(96)]
    vx = torch.from_numpy(np.stack([v[0] for v in val]).transpose(0, 3, 1, 2).copy())
    vt = torch.from_numpy(np.stack([v[1] for v in val]).transpose(0, 3, 1, 2).copy())
    it = batches(10)
    t0 = time.time(); run_l = 0; best = 1e9
    log = open(os.path.join(run, 'log.txt'), 'a')
    for step in range(1, steps + 1):
        A, T = next(it)
        x, t = compose(A, T, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = net(x)
        loss, _ = loss_fn(p.float(), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run_l += loss.item()
        if step % 200 == 0:
            msg = f'step {step} loss {run_l / 200:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s'
            print(msg, flush=True); log.write(msg + '\n'); log.flush(); run_l = 0
        if step % 1000 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                vl = []; vl1 = []
                for i in range(0, len(vx), 16):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        p = net(vx[i:i + 16].to(dev).to(memory_format=torch.channels_last))
                    a, b = loss_fn(p.float(), vt[i:i + 16].to(dev)); vl.append(a.item()); vl1.append(b.item())
            v = float(np.mean(vl))
            msg = f'VAL step {step} loss {v:.4f} logL1 {np.mean(vl1):.4f}'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
            torch.save(net.state_dict(), os.path.join(run, 'last.pt'))
            if v < best:
                best = v; torch.save(net.state_dict(), os.path.join(run, 'best.pt'))
            if step % 5000 == 0 or step == steps:
                real_sheet(net, dev, os.path.join(run, f'real_{step:06d}.jpg'))
            net.train()


if __name__ == '__main__':
    main()
