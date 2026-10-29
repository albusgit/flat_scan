"""feeder.py: the GPU generator (gen.py) in a process of its own, on a GPU of its own when the box has one spare.

In the trainers the generator ran inline: on the RTX 5060 making the page finder's 8 fresh photos took half of every
step (3.2 steps/s against 6.3 with no fresh photos). Here a child process makes batches on `dev` (another GPU, or the
trainer's own: the two then share it) into shared CPU slots; the trainer takes whatever is finished, never waiting
after the first fill. The training maths stay as they were; only how many fresh samples land per step changes (as many
as the generator manages, instead of a fixed number).

    f = Feeder('pf', 8, 'cuda:1', seed, pages_dir=..., bg_pack=...)    # mode 'pf' (GpuGen.batch) or 'sdhr' (sdhr_batch)
    for b in f.take(dev): ...   # finished batches (dicts of tensors, on dev), none waiting: an empty list
    b = f.take_one(dev)         # one batch, waiting for it (the ring's first fill)
    f.close()
"""
import os
import queue
import sys

import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE_PF = 384


def _layout(mode, k):
    """the slot tensors: name -> (shape, dtype), as the rings store them"""
    if mode == 'pf':
        return dict(x=((k, 3, SIZE_PF, SIZE_PF), torch.uint8), m=((k, 1, SIZE_PF, SIZE_PF), torch.bool),
                    gut=((k, 1, SIZE_PF, SIZE_PF), torch.bool), box=((k, 4), torch.long))
    return dict(x=((k, 3, 1440, 1024), torch.uint8), t=((k, 2, 121, 177), torch.float32), kind=((k,), torch.long),
                fg=((k, 121, 177), torch.uint8))


def _pack(mode, b):
    if mode == 'pf':
        return dict(x=(b['x'] * 255).round().to(torch.uint8), m=b['m'] > 0.5, gut=b['gut'] > 0.5, box=b['box'])
    return dict(x=b['x'], t=b['t'], kind=b['kind'], fg=b['fg'])


def _run(mode, k, dev, seed, kw, slots, free, ready, threads):
    torch.set_num_threads(threads)
    sys.path.insert(0, HERE)
    from gen import GpuGen
    if mode == 'sdhr':
        kw = dict(kw, mode='sdhr')
    g = GpuGen(seed=seed, dev=dev, **kw)
    make = g.batch if mode == 'pf' else g.sdhr_batch
    while True:
        s = free.get()
        if s is None:
            return
        b = _pack(mode, make(k))
        for name, t in slots[s].items():
            t.copy_(b[name])
        ready.put(s)


class Feeder:
    def __init__(self, mode, k, dev, seed, n_slots=6, threads=4, **kw):
        assert mode in ('pf', 'sdhr')
        self.mode, self.k = mode, k
        self.slots = [{name: torch.empty(shape, dtype=dt).share_memory_() for name, (shape, dt) in _layout(mode, k).items()}
                      for _ in range(n_slots)]
        ctx = mp.get_context('spawn')       # CUDA in the child: never fork
        self.free, self.ready = ctx.Queue(), ctx.Queue()
        for s in range(n_slots):
            self.free.put(s)
        self.proc = ctx.Process(target=_run, args=(mode, k, str(dev), seed, kw, self.slots, self.free, self.ready, threads),
                                daemon=True)
        self.proc.start()

    def _get(self, block, to):
        if not self.proc.is_alive():
            raise RuntimeError(f'the generator process stopped (exit code {self.proc.exitcode})')
        s = self.ready.get(timeout=600) if block else self.ready.get_nowait()
        b = {name: t.to(to, copy=True) for name, t in self.slots[s].items()}    # a plain (blocking) copy: done before the slot is reused
        self.free.put(s)
        return b

    def take_one(self, to='cpu'):
        return self._get(True, to)

    def take(self, to='cpu', most=4):
        out = []
        while len(out) < most:
            try:
                out.append(self._get(False, to))
            except queue.Empty:
                break
        return out

    def close(self):
        if self.proc.is_alive():
            self.free.put(None)
            self.proc.join(10)
            if self.proc.is_alive():
                self.proc.terminate()
