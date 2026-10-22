"""A pool of synthetic samples for training, as memory-mapped arrays (uint8):
  OUT_img.npy (N, S, S, 3) BGR, OUT_mask.npy (N, 2, S, S) rule masks x 255
  python gen_pool.py OUT N [SEED] [S]     (S = 400; the trainer crops 384)"""
import os, random, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth

out, n = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
S = int(sys.argv[4]) if len(sys.argv) > 4 else 400


def init():
    import cv2
    cv2.setNumThreads(1)


def work(i):
    rng = random.Random(seed * 1000003 + i)
    im, m = synth.sample(rng, S)
    return i, im, (m * 255 + 0.5).astype(np.uint8)


if __name__ == '__main__':
    imgs = np.lib.format.open_memmap(out + '_img.npy', 'w+', np.uint8, (n, S, S, 3))
    masks = np.lib.format.open_memmap(out + '_mask.npy', 'w+', np.uint8, (n, 2, S, S))
    with Pool(os.cpu_count(), initializer=init) as p:
        for k, (i, im, m) in enumerate(p.imap_unordered(work, range(n), chunksize=8)):
            imgs[i] = im; masks[i] = m
            if k % 500 == 0:
                print(k, flush=True)
    imgs.flush(); masks.flush()
    print('done', n)
