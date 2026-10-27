"""Shared training machinery for the FlatScan nets (tools/train/*): the fast settings every trainer uses.

  setup(seed)            TF32 matmuls / convs, cuDNN autotune, seeds; returns the device
  compiled(net)          torch.compile (falls back to eager when Triton is missing or the compile fails)
  adamw(params, lr, wd)  fused AdamW (foreach when fused is unavailable)
  onecycle(opt, lr, steps, warm)  linear warm-up then cosine to 2 % (OneCycle without momentum cycling)
  Ema(net, decay)        weights averaged on the GPU (lerp); .model is what is evaluated and exported
  bf16()                 autocast context in bfloat16 (no GradScaler needed)
  Log(path)              jsonl log + console line
  data_root()            FS_DATA, else E:/fs_datasets on Windows, /data elsewhere

Everything runs on one GPU; with several (torchrun), wrap with ddp() and use rank() / world()."""
import copy, json, math, os, random, sys, time

import numpy as np
import torch


def data_root():
    if os.environ.get('FS_DATA'):
        return os.environ['FS_DATA']
    return 'E:/fs_datasets' if os.name == 'nt' else '/data'


def setup(seed=0):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    random.seed(seed + rank()); np.random.seed(seed + rank()); torch.manual_seed(seed + rank())
    if world() > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group('nccl')
    dev = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(dev)
    return dev


def rank():
    return int(os.environ.get('RANK', 0))


def world():
    return int(os.environ.get('WORLD_SIZE', 1))


def ddp(net, dev):
    if world() == 1:
        return net
    return torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index])


def compiled(net, enable=True):
    if not enable or os.environ.get('FS_NO_COMPILE'):
        return net
    try:
        return torch.compile(net)
    except Exception as e:   # no Triton (some Windows installs), old GPU
        print('torch.compile unavailable, eager:', e, flush=True)
        return net


def adamw(params, lr, wd=1e-4):
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, foreach=True)


def onecycle(opt, lr, steps, warm):
    warm = max(1, warm)

    def f(s):
        if s < warm:
            return (s + 1) / warm
        t = (s - warm) / max(1, steps - warm)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    for g in opt.param_groups:
        g['lr'] = lr
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


class Ema:
    """exponential moving average of the weights (buffers copied), decay ramping up as the run starts"""

    def __init__(self, net, decay=0.999):
        self.model = copy.deepcopy(unwrap(net)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, net):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        src = unwrap(net)
        for e, p in zip(self.model.parameters(), src.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), src.buffers()):
            e.copy_(b)


def unwrap(net):
    net = getattr(net, '_orig_mod', net)          # torch.compile
    return getattr(net, 'module', net)            # DDP


def bf16():
    return torch.autocast('cuda', dtype=torch.bfloat16)


class Log:
    def __init__(self, path):
        self.f = open(path, 'a') if rank() == 0 else None
        self.t0 = time.time()

    def __call__(self, **kv):
        if self.f is None:
            return
        kv['sec'] = round(time.time() - self.t0, 1)
        self.f.write(json.dumps(kv) + '\n'); self.f.flush()
        print(' '.join(f'{k} {v:.5g}' if isinstance(v, float) else f'{k} {v}' for k, v in kv.items()), flush=True)
