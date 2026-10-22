"""A pool of synthetic samples for training, as memory-mapped arrays (uint8):
  OUT_img.npy (N, S, S, 3) BGR, OUT_mask.npy (N, 2, S, S) rule masks x 255,
  with synth2 also OUT_edge.npy (N, S, S): straight edges of pictures / dark bars (bit 1 horizontal, bit 2 vertical)
  python gen_pool.py OUT N [SEED] [S]     (S = 400; the trainer crops 384)
RN_GEN=synth2 draws with the round 11 generator (default synth); RN_PROCS worker processes (default all cores)."""
import os, random, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
synth = importlib.import_module(os.environ.get('RN_GEN', 'synth'))

out, n = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
S = int(sys.argv[4]) if len(sys.argv) > 4 else 400


def init():
    import cv2
    cv2.setNumThreads(1)


def work(i):
    rng = random.Random(seed * 1000003 + i)
    r = synth.sample(rng, S)
    return i, r[0], (r[1] * 255 + 0.5).astype(np.uint8), (r[2] if len(r) > 2 else None)


if __name__ == '__main__':
    imgs = np.lib.format.open_memmap(out + '_img.npy', 'w+', np.uint8, (n, S, S, 3))
    masks = np.lib.format.open_memmap(out + '_mask.npy', 'w+', np.uint8, (n, 2, S, S))
    edges = np.lib.format.open_memmap(out + '_edge.npy', 'w+', np.uint8, (n, S, S)) if hasattr(synth, 'picture') else None
    with Pool(int(os.environ.get('RN_PROCS', os.cpu_count())), initializer=init) as p:
        for k, (i, im, m, e) in enumerate(p.imap_unordered(work, range(n), chunksize=8)):
            imgs[i] = im; masks[i] = m
            if edges is not None: edges[i] = e
            if k % 500 == 0:
                print(k, flush=True)
    imgs.flush(); masks.flush()
    if edges is not None: edges.flush()
    print('done', n)
