"""Training data for the shadow net, made once before training (the trainer then only pairs and
augments it on the GPU, so the CPU never holds the GPU back).

  python make_bank.py pages OUT N [WORKERS]   N flat pages from the round-4 page generator -> OUT/pages/NNNNN.jpg
  python make_bank.py bank OUT N [WORKERS]    N page crops (with colour content) and N dark shading maps:
                                              OUT/albedo.npy  uint8   N x 512 x 384 x 3
                                              OUT/shade.npy   float16 N x 256 x 192 x 3 (half size, target scale)
Needs FS_DATA (default E:/fs_datasets on Windows, /data elsewhere) with round4/seg, round4/assets/{gfonts,bsds,scans}."""
import os, random, sys
from multiprocessing import Pool

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import cv2

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from fast import data_root  # noqa: E402

ROOT = data_root()
H, W = 512, 384
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'train_shade')


def page(i):
    os.environ['PACK_ROOT'] = os.path.join(ROOT, 'round4')
    sys.path.insert(0, os.path.join(ROOT, 'round4', 'seg'))
    import doc_gen
    p = os.path.join(OUT, 'pages', f'{i:05d}.jpg')
    if os.path.exists(p):
        return 0
    random.seed(1000003 * i + 17)   # doc_gen picks some fonts with the global random
    try:
        im, _ = doc_gen.page(random.Random(1000003 * i + 17), 900)
        cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return 1
    except Exception as e:
        print(i, e, flush=True)
        return 0


_ready = False


def item(i):
    global _ready
    import synth_dark
    if not _ready:
        synth_dark.configure(ROOT, os.path.join(OUT, 'pages')); _ready = True
    rng = random.Random(7919 * i + 3)
    a = (synth_dark.synth.page_albedo(rng, H, W) * 255 + 0.5).astype(np.uint8)
    L, _ = synth_dark.shading(rng, H // 2, W // 2)
    ref = np.percentile(cv2.resize(L, (W // 16, H // 16), interpolation=cv2.INTER_AREA).reshape(-1, 3), 99.5, axis=0)
    return i, a, np.clip(L / ref, 0.02, 1.0).astype(np.float16)


def main():
    stage, n = sys.argv[1], int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    os.makedirs(os.path.join(OUT, 'pages'), exist_ok=True)
    with Pool(workers) as pool:
        if stage == 'pages':
            done = 0
            for k, r in enumerate(pool.imap_unordered(page, range(n), chunksize=8)):
                done += r
                if k % 500 == 0:
                    print(f'pages {k}/{n}', flush=True)
            print(f'pages done: {done} new', flush=True)
            return
        A = np.lib.format.open_memmap(os.path.join(OUT, 'albedo.npy'), 'w+', np.uint8, (n, H, W, 3))
        T = np.lib.format.open_memmap(os.path.join(OUT, 'shade.npy'), 'w+', np.float16, (n, H // 2, W // 2, 3))
        for k, (i, a, t) in enumerate(pool.imap_unordered(item, range(n), chunksize=16)):
            A[i], T[i] = a, t
            if k % 1000 == 0:
                print(f'bank {k}/{n}', flush=True)
        A.flush(); T.flush()
        print('bank done', flush=True)


if __name__ == '__main__':
    main()
