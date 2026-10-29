"""sdhr_bench.py: SD-HR forward + backward speed on this GPU for micro-batch sizes and settings (round 19 bake-off).
Same random inputs and targets per variant; the loss over the timed steps is printed to check a setting does not
change the numbers. Variants: micro 2 / 4 / 6, channels_last, cudnn.benchmark, torch.compile of the encoder."""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'flatnet'))
import common as C  # noqa: E402
from run2_core import bm_forward, mixed_loss  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.92)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run(micro, cl, comp, steps=20, warm=5):
    torch.manual_seed(0)
    net = C.load_sdhr('E:/fs_flat2/run2b/step015000.pt', 'cuda').train()
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    if cl:
        net = net.to(memory_format=torch.channels_last)
    if comp:
        for n in ('enc_s1', 'enc_s2', 'enc_s3', 'enc_s4', 'enc_s5'):
            setattr(net, n, torch.compile(getattr(net, n)))
    params = [p for n, p in net.named_parameters() if not n.startswith('shd_')]
    opt = torch.optim.AdamW(params, 1e-5, fused=True)
    g = torch.Generator(device='cuda').manual_seed(1)
    x = torch.rand(micro, 3, 1440, 1024, device='cuda', generator=g)
    if cl:
        x = x.contiguous(memory_format=torch.channels_last)
    t = torch.rand(micro, 2, 121, 177, device='cuda', generator=g) * 2 - 1
    kind = torch.zeros(micro, dtype=torch.long, device='cuda')
    anc = torch.zeros(micro, dtype=torch.bool, device='cuda')
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for s in range(steps + warm):
        if s == warm:
            torch.cuda.synchronize(); t0 = time.time()
        p = bm_forward(net, x)
        loss, _ = mixed_loss(p, t, kind, anc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s >= warm:
            losses.append(loss.detach())
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    return dict(micro=micro, channels_last=cl, compile=comp, ms_per_sample=round(dt / micro * 1000, 1),
                peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), loss=round(float(torch.stack(losses).mean()), 5))


if __name__ == '__main__':
    for micro, cl, comp in [(2, 0, 0), (4, 0, 0), (6, 0, 0), (4, 1, 0), (4, 0, 1), (4, 1, 1)]:
        try:
            print(json.dumps(run(micro, cl, comp)), flush=True)
        except torch.OutOfMemoryError:
            print(json.dumps(dict(micro=micro, channels_last=cl, compile=comp, result='out of memory')), flush=True)
        torch.cuda.empty_cache()
