"""Fine-tune the shadow net (ShadeNet) for dark cast shadows.

  python train.py --bank DIR --out RUN [--init ONNX|scratch] [--steps 30000] [--bs 32] [--lr 4e-4] [--floor 0.12]
                  [--sizes 256x192:0.6,384x288:0.25,512x384:0.15] [--patience 6]

Progressive resizing (--sizes): the net is fully convolutional, and at 256 x 192 it trains 4x as many samples a
second as at the app's 512 x 384 (measured), so most of the run is at the small size, the last part at the real one.
The batch grows as the size shrinks (the same pixels per step); the maps are resampled to each size.
Early stopping (--patience): the run ends when the real-photo score has not improved for that many evaluations.

Data: make_bank.py's albedo.npy (page crops) and shade.npy (shading maps), held in RAM. Every step pairs
random pages with random maps on the GPU (so a bank of N gives N^2 pairings), augments both (flips,
half turns, the map zoomed 1-1.6x so shadows vary in size, colour cast, exposure, camera blur, shot noise,
8-bit levels) and trains on the photo = page x map. Nothing waits on the CPU.

Model: the app's net, its weights read from the shipped ONNX (--init, BN folded into the convs, so the
fine-tune starts exactly where the app is), or from scratch with BN (--init scratch).

Speed: bf16 autocast, channels_last, torch.compile, fused AdamW, TF32, EMA, warm-up + cosine; batches
staged in pinned memory one step ahead.

Selection: every --eval_every steps the EMA weights are scored on the real shadow photos with ground
truth (realval.py: jung2018, osr2020, RDD test) with the map clamped at --floor, as the app will use
it; best.pt is the best real score, not the best synthetic loss. log.jsonl has both."""
import argparse, os, sys, threading, queue

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import fast  # noqa: E402
import realval  # noqa: E402
from net import fused, from_onnx  # noqa: E402

H, W = 512, 384


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--init', default=os.path.join(HERE, '..', '..', '..', 'app', 'src', 'main', 'assets', 'models', 'shadenet.onnx'))
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=None, help='default 4e-4 (fine-tune) / 2e-3 (scratch)')
    p.add_argument('--warm', type=int, default=500)
    p.add_argument('--floor', type=float, default=0.12, help='the map clamp the app will use with this net')
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--val_every', type=int, default=3, help='every n-th pair of jung / osr (RDD: every 3rd)')
    p.add_argument('--shadow_w', type=float, default=2.0, help='extra loss weight in the shade (1 - map)')
    p.add_argument('--sizes', default='256x192:0.6,384x288:0.25,512x384:0.15', help='HxW:share of the steps, in order')
    p.add_argument('--patience', type=int, default=6, help='stop after this many evaluations without a better real score (0: never)')
    p.add_argument('--compile', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


class Bank:
    """the bank in RAM (pinned), batches gathered by a thread one step ahead"""

    def __init__(self, d, bs, dev):
        self.A = torch.from_numpy(np.load(os.path.join(d, 'albedo.npy')))
        self.T = torch.from_numpy(np.load(os.path.join(d, 'shade.npy')))
        self.n, self.bs, self.dev = len(self.A), bs, dev
        self.q = queue.Queue(maxsize=3)
        threading.Thread(target=self._fill, daemon=True).start()

    def _fill(self):
        g = torch.Generator().manual_seed(fast.rank())
        while True:
            ia = torch.randint(self.n, (self.bs,), generator=g)
            it = torch.randint(len(self.T), (self.bs,), generator=g)
            self.q.put((self.A[ia].pin_memory(), self.T[it].pin_memory()))

    def next(self):
        a, t = self.q.get()
        return a.to(self.dev, non_blocking=True), t.to(self.dev, non_blocking=True)


def blur(x, sigma):
    k = 2 * int(3 * sigma) + 1
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (r / sigma) ** 2); g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='replicate'), g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode='replicate'), g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)


def compose(a, t, hw=(H, W)):
    """photo = page x shading x cast x exposure, then camera, all on the GPU, at hw. a uint8 BHWC, t f16 BHWC (half size)"""
    B = a.shape[0]
    H, W = hw
    a = a.permute(0, 3, 1, 2).float().div_(255)
    if (a.shape[2], a.shape[3]) != (H, W):
        a = F.interpolate(a, size=(H, W), mode='area')
    t = t.permute(0, 3, 1, 2).float()
    # the map zoomed 1-1.6x (shadow size), flipped and turned on its own; the page flipped
    z = torch.empty(B, device=a.device).uniform_(1.0, 1.6)
    theta = torch.zeros(B, 2, 3, device=a.device)
    sx = torch.where(torch.rand(B, device=a.device) < 0.5, -1.0, 1.0); sy = torch.where(torch.rand(B, device=a.device) < 0.5, -1.0, 1.0)
    theta[:, 0, 0] = sx / z; theta[:, 1, 1] = sy / z
    theta[:, 0, 2] = (torch.rand(B, device=a.device) * 2 - 1) * (1 - 1 / z)
    theta[:, 1, 2] = (torch.rand(B, device=a.device) * 2 - 1) * (1 - 1 / z)
    grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
    t = F.grid_sample(t, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    flip = torch.rand(B, device=a.device) < 0.5
    a = torch.where(flip[:, None, None, None], a.flip(3), a)
    cast = torch.empty(B, 3, 1, 1, device=a.device).uniform_(0.9, 1.04)
    expo = torch.empty(B, 1, 1, 1, device=a.device).uniform_(0.75, 1.15)
    o = a * t * cast * expo
    # camera / resampling blur: four sigma groups (a separable conv each)
    q = B // 4
    parts = [o[:q]] + [blur(o[i * q:(i + 1) * q if i < 3 else B], s) for i, s in zip(range(1, 4), (0.4, 0.7, 1.1))]
    o = torch.cat(parts).clamp_(0, 1)
    sig = torch.empty(B, 1, 1, 1, device=a.device).uniform_(0.003, 0.03) * (torch.rand(B, 1, 1, 1, device=a.device) < 0.8)
    o = (o + torch.randn_like(o) * sig * o.clamp(min=0.02).sqrt()).clamp_(0, 1)
    o = torch.round(o * 255) / 255
    return o.contiguous(memory_format=torch.channels_last), t


def loss_fn(p, t, shadow_w):
    lp, lt = torch.log(p + 0.02), torch.log(t + 0.02)
    w = 1 + shadow_w * (1 - t.mean(1, keepdim=True)).clamp(min=0)
    l1 = ((lp - lt).abs() * w).mean() / w.mean()
    gy, gx = lp[..., 1:, :] - lp[..., :-1, :], lp[..., :, 1:] - lp[..., :, :-1]
    ty, tx = lt[..., 1:, :] - lt[..., :-1, :], lt[..., :, 1:] - lt[..., :, :-1]
    lg = (gy - ty).abs().mean() + (gx - tx).abs().mean()
    return l1 + 0.5 * lg, l1


def main():
    a = args()
    dev = fast.setup(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = fast.Log(os.path.join(a.out, 'log.jsonl'))
    if a.init == 'scratch':
        sys.path.insert(0, os.path.join(HERE, '..', '..', 'shadow_net'))
        from model import ShadeNet
        net, arch, lr = ShadeNet(), 'bn', a.lr or 2e-3
    else:
        net, arch, lr = from_onnx(a.init), 'fused', a.lr or 4e-4
    net = net.to(dev).to(memory_format=torch.channels_last)
    ema = fast.Ema(net, 0.999)
    model = fast.compiled(fast.ddp(net, dev), a.compile)
    opt = fast.adamw(net.parameters(), lr)
    sched = fast.onecycle(opt, lr, a.steps, a.warm)
    # the size schedule: (h, w, batch, last step)
    sizes, acc = [], 0.0
    for part in a.sizes.split(','):
        hw, share = part.split(':')
        h, w = (int(v) for v in hw.split('x'))
        acc += float(share)
        sizes.append((h, w, max(1, round(a.bs * (H * W) / (h * w))), int(round(acc * a.steps))))
    sizes[-1] = sizes[-1][:3] + (a.steps,)
    banks = {bs_: Bank(a.bank, bs_, dev) for bs_ in {s[2] for s in sizes}}
    pairs = realval.load(fast.data_root(), a.val_every) if fast.rank() == 0 else []
    if fast.rank() == 0 and not pairs:   # best.pt is picked on these: without them it would stay the starting net
        sys.exit(f'no real shadow pairs under {fast.data_root()} (realval.SETS): unpack the data first')

    def evaluate(step):
        m = ema.model

        def run(x):
            with torch.no_grad(), fast.bf16():
                return m(torch.from_numpy(x).to(dev).to(memory_format=torch.channels_last)).float().cpu().numpy()
        r = realval.score(pairs, run, a.floor)
        log(step=step, **{f'real_{k}': v for k, v in r.items()})
        return r['err']

    best = evaluate(0) if pairs else 1e9
    torch.save({'arch': arch, 'state': ema.model.state_dict(), 'step': 0, 'real_err': best}, os.path.join(a.out, 'best.pt'))
    run_l, since_best, si = 0.0, 0, 0
    for step in range(1, a.steps + 1):
        while step > sizes[si][3]:
            si += 1
        h, w, bs_, _ = sizes[si]
        x, t = compose(*banks[bs_].next(), (h, w))
        with fast.bf16():
            p = model(x)
        loss, l1 = loss_fn(p.float(), t, a.shadow_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); ema.update(net)
        run_l += loss.item() if step % 50 == 0 else 0.0
        if step % 500 == 0:
            log(step=step, loss=run_l / 10, lr=sched.get_last_lr()[0], size=f'{h}x{w}x{bs_}'); run_l = 0.0
        if fast.rank() == 0 and (step % a.eval_every == 0 or step == a.steps):
            err = evaluate(step)
            ck = {'arch': arch, 'state': ema.model.state_dict(), 'step': step, 'real_err': err}
            torch.save(ck, os.path.join(a.out, 'last.pt'))
            if err < best:
                best, since_best = err, 0
                torch.save(ck, os.path.join(a.out, 'best.pt'))
            else:
                since_best += 1
                if a.patience and since_best >= a.patience and si == len(sizes) - 1:
                    log(early_stop=step); break
    log(done=1, best_real_err=best)


if __name__ == '__main__':
    main()
