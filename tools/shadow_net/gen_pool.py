"""Flat pages for the shadow net's albedo pool, from the round-4 page generator (doc_gen.py).
  python gen_pool.py N [LONG_SIDE] [WORKERS]  -> C:/Users/User/Desktop/shadow_train/pool/NNNNN.jpg"""
import os, sys, random
from multiprocessing import Pool
sys.path.insert(0, r'C:\Users\User\Desktop\round4_robustness\seg')
OUT = r'C:\Users\User\Desktop\shadow_train\pool'


def one(i):
    import cv2, doc_gen
    p = os.path.join(OUT, f'{i:05d}.jpg')
    if os.path.exists(p):
        return
    rng = random.Random(1000003 * i + 17)
    try:
        im, _ = doc_gen.page(rng, LS)
        cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 93])
    except Exception as e:
        print(i, e, flush=True)


LS = int(sys.argv[2]) if len(sys.argv) > 2 else 900
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    n = int(sys.argv[1])
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 12) as pool:
        for k, _ in enumerate(pool.imap_unordered(one, range(n), chunksize=8)):
            if k % 500 == 0:
                print(k, flush=True)
