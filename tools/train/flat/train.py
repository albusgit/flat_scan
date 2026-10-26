"""Fine-tune UVDoc (the app's page-crop and first flattening net) on the UVDoc dataset's true backward grids.

  python train.py --data PREP_DIR --out RUN [--init best_model.pkl] [--steps 40000] [--bs 24] [--lr 1e-4]

Why this run is set up as it is (the Round 11 SD-HR fine-tune improved its own validation but lost end to end):
  - ground truth, not pseudo ground truth: UVDoc's captured 3D paper shapes, folds and creases included;
  - the app's input, not UVDoc's training crop: the page's box + a margin of U(0.02, 0.10) of its long side
    (the app: 0.06), corners jittered by 2 %, clamped to the photo, turned upright, area-resized to 488 x 712.
    Upstream trained on tight crops (0 - 20 px); the app has always fed it the wider crop;
  - selection on real photos run as the app runs the net (realval.py: two passes on dir300, MS-SSIM and a
    flow-length LD), never on the training data's own validation alone.
Loss: 5 x L1 on the grid + L1 between the photo unwarped by the predicted and by the true grid (upstream's
2D terms; the 3D head is not used by the app and gets no loss). In-plane rotation +-ROT deg moves the targets too.
Speed: bf16 autocast, channels_last, torch.compile, fused AdamW, TF32, EMA, warm-up + cosine; photometric
augmentation, blur, noise and rotation on the GPU; workers only decode and crop."""
import argparse, json, math, os, random, sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import fast  # noqa: E402
import realval  # noqa: E402
from uvdoc_model import UVDocnet  # noqa: E402

IN_H, IN_W = 712, 488


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--init', default=os.path.join(fast.data_root(), 'models', 'uvdoc_best_model.pkl'))
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--bs', type=int, default=24)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warm', type=int, default=500)
    p.add_argument('--rot', type=float, default=6.0)
    p.add_argument('--margin', default='0.02,0.10')
    p.add_argument('--workers', type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument('--eval_every', type=int, default=2000)
    p.add_argument('--val_set', default=os.path.join(fast.data_root(), 'sets', 'dir300.json'))
    p.add_argument('--val_every_n', type=int, default=3, help='every n-th photo of the validation set')
    p.add_argument('--compile', type=int, default=1)
    p.add_argument('--patience', type=int, default=5, help='stop after this many evaluations without a better MS-SSIM (0: never)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def app_crop(grid, w, h, margin, rng, jitter):
    """the box the app crops around the page (core/src/uvdoc.cpp uv_crop) and the turned flag"""
    c = np.array([grid[:, 0, 0], grid[:, 0, -1], grid[:, -1, -1], grid[:, -1, 0]])
    x0, y0 = c.min(0); x1, y1 = c.max(0)
    m = margin * max(x1 - x0, y1 - y0)
    j = jitter * max(x1 - x0, y1 - y0)
    d = rng.uniform(-j, j, 4) if j > 0 else np.zeros(4)
    bx0, by0 = max(0, int(np.floor(x0 - m + d[0]))), max(0, int(np.floor(y0 - m + d[1])))
    bx1, by1 = min(w, int(np.ceil(x1 + m + d[2])) + 1), min(h, int(np.ceil(y1 + m + d[3])) + 1)
    turned = np.linalg.norm(c[1] - c[0]) > np.linalg.norm(c[3] - c[0])
    return (bx0, by0, max(bx0 + 16, bx1), max(by0 + 16, by1)), bool(turned)


def sample(img, grid, box, turned):
    """net input uint8 HxWx3 RGB and its target grid 2 x 45 x 31 in [-1, 1] (UVDoc's normalisation)"""
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    g = grid.copy()
    g[0] -= x0; g[1] -= y0
    cw, ch = x1 - x0, y1 - y0
    if turned:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        g = np.stack([(ch - 1) - g[1], g[0]])
        cw, ch = ch, cw
    small = cv2.resize(crop, (IN_W, IN_H), interpolation=cv2.INTER_AREA if cw > IN_W else cv2.INTER_LINEAR)
    t = np.stack([g[0] / cw * 2 - 1, g[1] / ch * 2 - 1])[:, ::2, ::2]   # 89 x 61 -> 45 x 31, as upstream subsamples
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB), np.ascontiguousarray(t, dtype=np.float32)


class Data(IterableDataset):
    def __init__(self, root, items, grids, margins):
        super().__init__()
        self.root, self.items, self.grids, self.margins = root, items, grids, margins

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng((info.id if info else 0) * 7919 + fast.rank() * 104729 + random.randrange(1 << 20))
        while True:
            k = int(rng.integers(len(self.items)))
            img = cv2.imread(os.path.join(self.root, self.items[k]['img']))
            if img is None:
                continue
            grid = self.grids[self.items[k]['row']]
            box, turned = app_crop(grid, img.shape[1], img.shape[0], rng.uniform(*self.margins), rng, 0.02)
            x, t = sample(img, grid, box, turned)
            yield torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(t)


def gpu_aug(x, t, rot):
    """x uint8 B x 3 x H x W -> float [0, 1] with colour / exposure / gamma / blur / noise and an in-plane turn that moves t"""
    B, _, H, W = x.shape
    dev = x.device
    x = x.float().div_(255)
    u = lambda lo, hi, *s: torch.empty(*(s or (B,)), device=dev).uniform_(lo, hi)
    y = (x * u(0.7, 1.2).view(B, 1, 1, 1) + u(-0.08, 0.08).view(B, 1, 1, 1)).clamp(0, 1) ** u(0.75, 1.35).view(B, 1, 1, 1)
    gr = y.mean(1, keepdim=True)
    y = gr + (y - gr) * u(0.6, 1.3).view(B, 1, 1, 1)
    y = y * u(0.9, 1.1, B, 3).view(B, 3, 1, 1)
    on = (torch.rand(B, 1, 1, 1, device=dev) < 0.8).float()
    x = (on * y + (1 - on) * x).clamp(0, 1)
    q = B // 3
    if q:
        k = torch.tensor([1, 4, 6, 4, 1], device=dev, dtype=x.dtype); k = k / k.sum()
        b = x[:q]
        b = F.conv2d(F.pad(b, (2, 2, 0, 0), mode='replicate'), k.view(1, 1, 1, 5).expand(3, 1, 1, 5), groups=3)
        b = F.conv2d(F.pad(b, (0, 0, 2, 2), mode='replicate'), k.view(1, 1, 5, 1).expand(3, 1, 5, 1), groups=3)
        x = torch.cat([b, x[q:]])
    x = (x + torch.randn_like(x) * u(0, 0.02).view(B, 1, 1, 1) * (torch.rand(B, 1, 1, 1, device=dev) < 0.3)).clamp(0, 1)
    if rot > 0:
        ang = u(-rot, rot) * math.pi / 180 * (torch.rand(B, device=dev) < 0.5).float()
        ca, sa = torch.cos(ang).view(B, 1, 1), torch.sin(ang).view(B, 1, 1)
        cx, cy = (W - 1) / 2, (H - 1) / 2
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        px = ca * (xx - cx) + sa * (yy - cy) + cx
        py = -sa * (xx - cx) + ca * (yy - cy) + cy
        x = F.grid_sample(x, torch.stack([px / (W - 1) * 2 - 1, py / (H - 1) * 2 - 1], -1), mode='bilinear', padding_mode='reflection', align_corners=True)
        tx, ty = (t[:, 0] + 1) / 2 * W - cx, (t[:, 1] + 1) / 2 * H - cy
        t = torch.stack([((ca * tx - sa * ty) + cx) / W * 2 - 1, ((sa * tx + ca * ty) + cy) / H * 2 - 1], 1)
    return x.contiguous(memory_format=torch.channels_last), t


def unwarp(x, g):
    gg = F.interpolate(g, size=(IN_H, IN_W), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
    return F.grid_sample(x, gg, align_corners=True)


def main():
    a = args()
    dev = fast.setup(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = fast.Log(os.path.join(a.out, 'log.jsonl'))
    items = json.load(open(os.path.join(a.data, 'index.json')))
    for k, it in enumerate(items):
        it['row'] = k
    grids = np.load(os.path.join(a.data, 'grids.npy'))
    train_items = [it for it in items if not it['val']]
    held = [it for it in items if it['val']]
    net = UVDocnet(num_filter=32, kernel_size=5)
    ck = torch.load(a.init, map_location='cpu', weights_only=False)
    net.load_state_dict(ck.get('model_state', ck))
    net = net.to(dev).to(memory_format=torch.channels_last)
    ema = fast.Ema(net, 0.999)
    model = fast.compiled(fast.ddp(net, dev), a.compile)
    opt = fast.adamw(net.parameters(), a.lr)
    sched = fast.onecycle(opt, a.lr, a.steps, a.warm)
    margins = tuple(float(v) for v in a.margin.split(','))
    dl = iter(DataLoader(Data(a.data, train_items, grids, margins), batch_size=a.bs, num_workers=a.workers,
                         pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None))
    vitems = realval.load(a.val_set, a.val_every_n, fast.data_root()) if fast.rank() == 0 else []
    # the held-out geometries at the app's 6 % margin (no jitter): the dataset's own check
    hrng = np.random.default_rng(1)
    hx, ht = [], []
    for it in held[:400]:
        img = cv2.imread(os.path.join(a.data, it['img']))
        box, turned = app_crop(grids[it['row']], img.shape[1], img.shape[0], 0.06, hrng, 0.0)
        x, t = sample(img, grids[it['row']], box, turned)
        hx.append(x); ht.append(t)

    def evaluate(step):
        m = ema.model.eval()
        r = {}
        if hx:
            l = []
            with torch.no_grad(), fast.bf16():
                for i in range(0, len(hx), 16):
                    x = torch.from_numpy(np.stack(hx[i:i + 16])).permute(0, 3, 1, 2).to(dev).float().div(255).contiguous(memory_format=torch.channels_last)
                    l.append(F.l1_loss(m(x)[0].float(), torch.from_numpy(np.stack(ht[i:i + 16])).to(dev)).item())
            r['held_l1'] = float(np.mean(l))

        def grid_fn(x):
            with torch.no_grad(), fast.bf16():
                return m(torch.from_numpy(x).to(dev).contiguous(memory_format=torch.channels_last))[0][0].float().cpu().numpy()
        r.update(realval.score(vitems, grid_fn))
        log(step=step, **r)
        return r

    r0 = evaluate(0)
    best = r0['ms_ssim']
    torch.save({'state': ema.model.state_dict(), 'step': 0, 'real': r0}, os.path.join(a.out, 'best.pt'))
    run, since_best = 0.0, 0
    for step in range(1, a.steps + 1):
        x, t = next(dl)
        x, t = gpu_aug(x.to(dev, non_blocking=True), t.to(dev, non_blocking=True), a.rot)
        with fast.bf16():
            g = model(x)[0]
        g = g.float()
        loss = 5 * F.l1_loss(g, t) + F.l1_loss(unwarp(x, g), unwarp(x, t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(net)
        if step % 50 == 0:
            run += loss.item()
        if step % 500 == 0:
            log(step=step, loss=run / 10, lr=sched.get_last_lr()[0]); run = 0.0
        if fast.rank() == 0 and (step % a.eval_every == 0 or step == a.steps):
            r = evaluate(step)
            ck = {'state': ema.model.state_dict(), 'step': step, 'real': r}
            torch.save(ck, os.path.join(a.out, 'last.pt'))
            if r['ms_ssim'] > best:
                best, since_best = r['ms_ssim'], 0
                torch.save(ck, os.path.join(a.out, 'best.pt'))
            else:
                since_best += 1
                if a.patience and since_best >= a.patience:
                    log(early_stop=step); break
    log(done=1, best_ms_ssim=best)


if __name__ == '__main__':
    main()
