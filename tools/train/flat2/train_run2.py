"""train_run2.py: SD-HR fine-tune, run 2 (round 18). What changed from train_sdhr.py main (run 1), and why:
- the coordinate head in float32 (run2_core.bm_forward): run 1's bf16 head made most of the derivative loss rounding;
- exact source counts per batch (--counts synth,uvdoc,anchor; run 1 drew sources at random: 17 % of batches had no
  anchor), each sample drawn by its (source, id), so workers are persistent and a resumed run draws the same;
- the teacher from --teacher_init (the shipped weights), not from --init: a run started from a fine-tuned checkpoint
  is still anchored to the original;
- losses: run2_core.mixed_loss (anchors: their own lighter objective; a cell-area floor in place of the fold term);
- encoder and head at their own learning rates, warmup then cosine to 10 %; BatchNorm statistics and affine
  parameters frozen;
- the effective batch over micro-batches (--micro), checkpoints with the optimizer, scheduler, RNG and EMA state
  (--resume), the synthetic set fixed at launch (its index is read once);
- the net-only real check (RealVal) and the held synthetic check in float32, grids resized as the app reads them.
The checkpoint chosen here (best_proxy.pt) is net-only evidence: the app pipeline decides."""
import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'flatnet'))
sys.path.insert(0, os.path.join(HERE, '..', 'flat'))
import fast  # noqa: E402
import common as C  # noqa: E402
from run2_core import BalancedBatches, bm_forward, mixed_loss  # noqa: E402
from train_sdhr import Mixed, RealVal, gpu_aug, synth_eval  # noqa: E402

SOURCES = ('synth', 'uvdoc', 'anchor')


def worker_init(_):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def atomic_save(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


class GenRing:
    """synthetic SD-HR samples made on the GPU (tools/train/gpugen, mode sdhr) in a ring of `size`: `new` fresh ones per
    step, `per_step` drawn from the ring (each used ~ size x per_step / new times over its life, augmented afresh)"""

    def __init__(self, size, new, per_step, seed, dev, pages, bg_pack, gen_dev=''):
        sys.path.insert(0, os.path.join(HERE, '..', 'gpugen'))
        if gen_dev:     # the generator in its own process (gpugen/feeder.py): fresh samples as fast as it makes them
            from feeder import Feeder
            self.f, self.g = Feeder('sdhr', max(4, new), gen_dev, seed, pages_dir=pages, bg_pack=bg_pack), None
        else:
            from gen import GpuGen
            self.f, self.g = None, GpuGen(mode='sdhr', seed=seed, dev=dev, pages_dir=pages, bg_pack=bg_pack)
        self.size, self.new, self.per_step, self.dev = size, new, per_step, dev
        self.x = torch.zeros(size, 3, 1440, 1024, dtype=torch.uint8, device=dev)
        self.t = torch.zeros(size, 2, 121, 177, device=dev)
        self.kind = torch.zeros(size, dtype=torch.long, device=dev)
        self.fg = torch.zeros(size, 121, 177, dtype=torch.uint8, device=dev)    # ink at the target's grid (gen.py)
        self.head, self.n, self.made = 0, 0, 0
        while self.n < size:
            if self.f:
                self.put(self.f.take_one(dev))
            else:
                self.add(min(16, size - self.n))
        self.rng = torch.Generator(device=dev)
        self.rng.manual_seed(seed)

    def add(self, k):
        self.put(self.g.sdhr_batch(k))

    def put(self, b):
        k = len(b['x'])
        idx = (self.head + torch.arange(k, device=self.dev)) % self.size
        self.x[idx], self.t[idx], self.kind[idx], self.fg[idx] = b['x'], b['t'], b['kind'], b['fg']
        self.head = (self.head + k) % self.size
        self.n = min(self.size, self.n + k)
        self.made += k

    def draw(self):
        if self.f:
            for b in self.f.take(self.dev):
                self.put(b)
        elif self.new:
            self.add(self.new)
        i = torch.randint(0, self.size, (self.per_step,), device=self.dev, generator=self.rng)
        return self.x[i], self.t[i], self.kind[i], self.fg[i].float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', required=True, help='folders of make_data.py samples (comma separated)')
    ap.add_argument('--held', required=True, help='a separately generated folder of held-out synthetic samples')
    ap.add_argument('--uvdoc', default='')
    ap.add_argument('--anchor', default='')
    ap.add_argument('--init', default=C.SD_CKPT)
    ap.add_argument('--teacher_init', default=C.SD_CKPT)
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume', default='')
    ap.add_argument('--counts', default='8,3,5', help='synth, uvdoc, anchor samples per optimizer step')
    ap.add_argument('--gpugen', type=int, default=0, help='also this many samples per step from the GPU generator (GenRing)')
    ap.add_argument('--gpugen_ring', type=int, default=384); ap.add_argument('--gpugen_new', type=int, default=4)
    ap.add_argument('--gen_dev', default='', help='the generator in its own process on this device (cuda:1: a GPU of its own): fresh samples as fast as it makes them, the step never waits for them')
    ap.add_argument('--pages', default='C:/fs_train/pages2'); ap.add_argument('--bg_pack', default='C:/fs_train/pf2/train_neg2.npz')
    ap.add_argument('--micro', type=int, default=8, help='samples per forward / backward')
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--lr', type=float, default=2e-5, help='bridge and head')
    ap.add_argument('--enc_lr', type=float, default=5e-6)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--eval', type=int, default=2000)
    ap.add_argument('--save', default='1000', help='also save at these steps (comma separated)')
    ap.add_argument('--held_n', type=int, default=800)
    ap.add_argument('--real_sets', default='dir300')
    ap.add_argument('--real_every', type=int, default=2)
    ap.add_argument('--uvdoc_onnx', default=os.path.join(HERE, '..', '..', '..', 'app', 'src', 'main', 'assets', 'models', 'uvdoc.onnx'))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--legacy_head', action='store_true', help='control only: the head under bf16, as run 1')
    ap.add_argument('--held_real', default='', help='held-out real flat photos (make_real.py --split val): logged as hr_*')
    ap.add_argument('--channels_last', type=int, default=1, help='round 19 bake-off: 6 %% faster, the same loss')
    ap.add_argument('--compile', type=int, default=1, help='torch.compile the encoder stages in place (checkpoint keys unchanged)')
    ap.add_argument('--ckpt', type=int, default=500, help='a rolling resume point (last.pt) every this many steps')
    ap.add_argument('--live_reload', type=float, default=0, help='seconds: workers re-read the synthetic index this often '
                    '(train while make_data.py still writes; the run is then not exactly resumable)')
    a = ap.parse_args()
    counts = dict(zip(SOURCES, (int(v) for v in a.counts.split(','))))
    batch = sum(counts.values()) + a.gpugen
    if len(a.counts.split(',')) != 3 or batch < 1 or a.steps <= a.warmup:
        ap.error('--counts: three counts; --steps more than --warmup')
    os.makedirs(a.out, exist_ok=True)
    dev = fast.setup(a.seed)
    torch.set_num_threads(2)
    tr = Mixed(a.synth.split(','), a.uvdoc, a.anchor, [max(v, 1e-9) for v in counts.values()], 0, seed=a.seed, all_ids=True)
    for s, items in (('synth', tr.synth), ('uvdoc', tr.uv_items), ('anchor', tr.anchor)):
        if counts[s] and not items:
            ap.error(f'{s}: asked for {counts[s]} per batch, but there are none')
    tr.live_reload = a.live_reload
    va = Mixed([a.held], '', '', [1, 0, 0], 0, held=True, all_ids=True)
    va.length = min(a.held_n, len(va.synth))
    hr = None
    if a.held_real:
        hr = Mixed([a.held_real], '', '', [1, 0, 0], 0, held=True, all_ids=True)
        hr.length = min(a.held_n, len(hr.synth))
    seen = {(os.path.abspath(r['root']), r['id']) for r in tr.synth}
    if any((os.path.abspath(r['root']), r['id']) in seen for r in va.synth):
        ap.error('--held overlaps --synth')
    with open(os.path.join(a.out, 'run.json'), 'w') as f:
        json.dump({'args': vars(a), 'argv': sys.argv, 'synth': len(tr.synth), 'uvdoc': len(tr.uv_items),
                   'anchor': len(tr.anchor), 'held': va.length}, f, indent=1)
    print(f'train: synth {len(tr.synth)}, uvdoc {len(tr.uv_items)}, anchor {len(tr.anchor)}; per batch {counts}; held {va.length}', flush=True)

    mf = torch.channels_last if a.channels_last else torch.contiguous_format
    net = C.load_sdhr(a.init, dev).to(memory_format=mf)
    teacher = None
    if counts['anchor']:
        teacher = C.load_sdhr(a.teacher_init, dev).eval().to(memory_format=mf)
        teacher.requires_grad_(False)
    bn = {id(p) for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for p in m.parameters()}
    enc, head = [], []
    for name, p in net.named_parameters():
        p.requires_grad_(not name.startswith('shd_') and id(p) not in bn)   # the shading decoder is not used
        if p.requires_grad:
            (enc if name.startswith('enc_') else head).append(p)
    params = enc + head
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.enc_lr}, {'params': head, 'lr': a.lr}], weight_decay=1e-4, fused=True)

    def lr_at(step):
        if step < a.warmup:
            return (step + 1) / a.warmup
        q = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * q))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = fast.Ema(net, 0.999)
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed + 1)
    log = fast.Log(os.path.join(a.out, 'log.jsonl'))
    vl = DataLoader(va, batch_size=a.micro, shuffle=False, num_workers=min(2, a.workers), worker_init_fn=worker_init)
    hl = DataLoader(hr, batch_size=a.micro, shuffle=False, num_workers=min(2, a.workers), worker_init_fn=worker_init) if hr else None
    real = RealVal(a.real_sets.split(','), a.real_every, a.uvdoc_onnx, fast.data_root()) if a.real_sets else None
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        for k in ('counts', 'micro', 'steps', 'lr', 'enc_lr', 'warmup', 'seed', 'teacher_init', 'legacy_head', 'synth'):
            if ck['args'][k] != vars(a)[k]:
                raise SystemExit(f'--resume: {k} differs from the run being resumed')
        net.load_state_dict(ck['student'])
        ema.model.load_state_dict(ck['state_dict'])
        ema.n = ck['ema_n']
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        gen.set_state(ck['rng']['aug'].cpu())
        torch.set_rng_state(ck['rng']['torch'].cpu())
        torch.cuda.set_rng_state(ck['rng']['cuda'].cpu())
        random.setstate(ck['rng']['python'])
        start, base_syn, base_real, best = ck['step'], ck['base_syn'], ck['base_real'], ck['best']
    else:
        base_syn = synth_eval(ema.model, vl, dev)
        base_real = real(ema.model, dev) if real else (0, 0)
        best = base_real[0]   # a checkpoint must beat the start
        hrs = synth_eval(ema.model, hl, dev) if hl else {}
        log(step=0, real_ms_ssim=base_real[0], real_ld=base_real[1], **{f'syn_{k}': v[0] for k, v in base_syn.items()},
            **{f'synd_{k}': v[1] for k, v in base_syn.items()}, **{f'hr_{k}': v[0] for k, v in hrs.items()},
            **{f'hrd_{k}': v[1] for k, v in hrs.items()})
    kw = dict(num_workers=a.workers, pin_memory=True, worker_init_fn=worker_init)
    if a.workers:
        kw.update(persistent_workers=True, prefetch_factor=2)
    tl = DataLoader(tr, batch_sampler=BalancedBatches(counts, a.steps, start, a.seed), **kw)
    saves = {int(v) for v in a.save.split(',') if v}
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()   # BatchNorm keeps the 1M-sample statistics
    if a.compile:   # after the EMA copy: only the student's encoder, in place
        for n in ('enc_s1', 'enc_s2', 'enc_s3', 'enc_s4', 'enc_s5'):
            getattr(net, n).compile()
    t0, acc, skipped = time.time(), {}, 0
    # the EMA in one fused update per step (fast.Ema's loop launched hundreds of small kernels)
    ema_p, net_p = list(ema.model.parameters()), list(net.parameters())
    ema_b, net_b = list(ema.model.buffers()), list(net.buffers())

    @torch.no_grad()
    def ema_update():
        ema.n += 1
        d = min(ema.decay, (1 + ema.n) / (10 + ema.n))
        torch._foreach_lerp_(ema_p, net_p, 1 - d)
        torch._foreach_copy_(ema_b, net_b)
    def make_ck(step, rl, syn):
        return {'state_dict': ema.model.state_dict(), 'student': net.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(), 'ema_n': ema.n, 'step': step, 'args': vars(a), 'real': rl,
                'synth': syn, 'base_syn': base_syn, 'base_real': base_real, 'best': best,
                'rng': {'aug': gen.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
                        'python': random.getstate()}}
    ring = GenRing(a.gpugen_ring, a.gpugen_new, a.gpugen, a.seed + 7919 * (1 + start), dev, a.pages, a.bg_pack, a.gen_dev) if a.gpugen else None
    for step, (x, t, kind, anc) in enumerate(tl, start + 1):
        x = x.to(dev, non_blocking=True)
        t = t.to(dev, non_blocking=True)
        kind, anc = kind.to(dev), anc.to(dev).bool()
        fg = torch.zeros(len(x), t.shape[-2], t.shape[-1], device=dev)          # (no ink known: loaded samples)
        if ring is not None:
            gx, gt, gk, gf = ring.draw()
            x, t, fg = torch.cat([x, gx]), torch.cat([t, gt]), torch.cat([fg, gf])
            kind, anc = torch.cat([kind, gk]), torch.cat([anc, torch.zeros_like(gk, dtype=torch.bool)])
        x = gpu_aug(x, gen).contiguous(memory_format=mf)
        if teacher is not None and anc.any():
            with torch.no_grad():
                t[anc] = bm_forward(teacher, x[anc])
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=dev)
        for lo in range(0, batch, a.micro):
            sl = slice(lo, min(batch, lo + a.micro))
            p = bm_forward(net, x[sl], fp32_head=not a.legacy_head)
            loss, q = mixed_loss(p, t[sl], kind[sl], anc[sl], fg[sl])
            (loss * (p.shape[0] / batch)).backward()
            total += loss.detach() * p.shape[0] / batch
            # per-source term sums stay on the GPU: read once every 100 steps (no wait for the GPU each step)
            src = torch.where(anc[sl], 2, torch.where(kind[sl] == Mixed.KINDS.index('uvdoc'), 1, 0))
            for si, sn in enumerate(SOURCES):
                m = (src == si).float()
                for k, v in q.items():
                    e = acc.setdefault(f'{sn}_{k}', [torch.zeros((), device=dev), torch.zeros((), device=dev)])
                    e[0] += (v.detach() * m).sum(); e[1] += m.sum()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):   # skip this step (BatchNorm is frozen: nothing else to undo)
            skipped += 1
            opt.zero_grad(set_to_none=True)
            print(f'step {step}: gradient not finite, skipped ({skipped})', flush=True)
            if skipped > 50:
                raise SystemExit('more than 50 non-finite steps: stopping')
        else:
            opt.step()
        sched.step()
        ema_update()
        if step % 100 == 0:
            if a.live_reload:
                tr.reload()   # the main process's count, for the log
            log(step=step, loss=total.item(), grad=float(gn), lr=sched.get_last_lr()[1], sec_per_step=(time.time() - t0) / (step - start), skipped=skipped,
                synth_n=len(tr.synth), made=ring.made if ring else 0, **{k: (v[0] / v[1].clamp_min(1)).item() for k, v in sorted(acc.items()) if v[1].item() > 0})
            acc = {}
        if step % a.eval == 0 or step == a.steps or step in saves:
            syn = synth_eval(ema.model, vl, dev)
            hrs = synth_eval(ema.model, hl, dev) if hl else {}
            rl = real(ema.model, dev) if real and (step % a.eval == 0 or step == a.steps) else None
            better = rl is not None and rl[0] > best
            if better:
                best = rl[0]
            log(step=step, real_ms_ssim=rl[0] if rl else -1, real_ld=rl[1] if rl else -1, best=best,
                **{f'syn_{k}': v[0] for k, v in syn.items()}, **{f'synd_{k}': v[1] for k, v in syn.items()},
                **{f'hr_{k}': v[0] for k, v in hrs.items()}, **{f'hrd_{k}': v[1] for k, v in hrs.items()})
            ck = make_ck(step, rl, syn)
            atomic_save(ck, os.path.join(a.out, f'step{step:06d}.pt'))
            if better:
                atomic_save({'state_dict': ck['state_dict'], 'step': step, 'real': rl, 'synth': syn}, os.path.join(a.out, 'best_proxy.pt'))
        elif step % a.ckpt == 0:   # a resume point between evaluations (overnight_r19.py resumes from the newest)
            atomic_save(make_ck(step, None, None), os.path.join(a.out, 'last.pt'))
            with open(os.path.join(a.out, 'last.json'), 'w') as f:
                json.dump({'step': step}, f)
    print('done; best net-only real MS-SSIM', best, 'start', base_real, flush=True)


if __name__ == '__main__':
    main()
