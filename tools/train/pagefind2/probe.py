"""where a page-finder step's time goes: data, teacher, forward/backward (bf16, channels_last, batch from argv)"""
import sys, time, numpy as np, torch
sys.argv = ['x', '--synth', 'C:/fs_train/pf2/synth_a.npz']
import train as T
from net import SegNet2
bs = int(__import__('os').environ.get('BS', 16))
torch.cuda.set_per_process_memory_fraction(0.9)   # fail fast instead of spilling into system RAM (Windows)
dev = 'cuda'
torch.backends.cudnn.benchmark = True
pk = T.GpuPack('C:/fs_train/pf2/synth_a.npz', dev)
net = SegNet2(T.IMAGENET).to(dev).to(memory_format=torch.channels_last)
sys.path.insert(0, T.ROUND4 + '/seg'); import segnet as old
teacher = old.SegNet().to(dev).eval(); teacher.load_state_dict(torch.load(T.TEACHER, map_location=dev))
opt = torch.optim.AdamW(net.parameters(), 1e-3, fused=True)
gen = torch.Generator(device=dev); gen.manual_seed(0); rng = np.random.default_rng(0)
acc = {'data': 0, 'teach': 0, 'aug': 0, 'fb': 0}
def tick(): torch.cuda.synchronize(); return time.time()
for step in range(25):
    if step == 0: print("start", torch.cuda.memory_allocated() / 1e9, flush=True)
    t0 = tick(); x, m, v, b = pk.get(rng.integers(0, pk.n, bs).tolist()); t1 = tick()
    gt = T.teacher_gutter(teacher, x, b); t2 = tick()
    x, m, v, gt = T.augment(x, m, v, gt, gen); e = T.edge_target(m, v); x = T.normalise(x).contiguous(memory_format=torch.channels_last); t3 = tick()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = net(x)
    loss, _ = T.loss_fn(out, m, v, e, gt); opt.zero_grad(); loss.backward(); opt.step(); t4 = tick()
    print(step, round(t4 - t0, 3), flush=True)
    if step >= 10:
        for k, d in zip(acc, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)): acc[k] += d
    if step == 12: print('peak GB', torch.cuda.max_memory_allocated() / 1e9, flush=True)
print({k: round(v / 15 * 1000, 1) for k, v in acc.items()}, 'ms per step')
