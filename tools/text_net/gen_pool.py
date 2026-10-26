"""A pool of synth_text samples as memory-mapped arrays (as tools/rule_net/gen_pool.py, 4 mask channels):
  OUT_img.npy (N, S, S, 3) BGR, OUT_mask.npy (N, 4, S, S) [rule h, rule v, text h, text v] x 255,
  OUT_edge.npy (N, S, S) picture edges (bit 1 horizontal, bit 2 vertical), OUT_script.json (each sample's script)
  python gen_pool.py CORPUS.json OUT N [SEED] [S]     (S = 400); TN_PROCS worker processes (default cores - 2);
  TN_LAYOUT=mag: magazine / advert pages instead (tools/train/rules/synth_mag.py)
Writes OUT.log ("done" at the end, as train_text.py waits for)."""
import json, os, random, sys
from multiprocessing import Pool
import numpy as np

corpus, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
S = int(sys.argv[5]) if len(sys.argv) > 5 else 400
os.environ['TN_CORPUS'] = corpus


def init():
    import cv2
    cv2.setNumThreads(1)


def work(i):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import synth_text
    layout = None
    if os.environ.get('TN_LAYOUT') == 'mag':   # magazine / advert pages (tools/train/rules/synth_mag.py)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train', 'rules'))
        import synth_mag
        layout = synth_mag.mag_layout
    rng = random.Random(seed * 1000003 + i)
    for k in range(5):   # a sample that fails (a font Qt cannot load) is drawn again
        try:
            im, m, e, s = synth_text.sample(rng, S, layout)
            return i, im, (m * 255 + 0.5).astype(np.uint8), e, s
        except Exception as ex:
            err = repr(ex)
    raise RuntimeError(err)


if __name__ == '__main__':
    log = open(out + '.log', 'w')
    imgs = np.lib.format.open_memmap(out + '_img.npy', 'w+', np.uint8, (n, S, S, 3))
    masks = np.lib.format.open_memmap(out + '_mask.npy', 'w+', np.uint8, (n, 4, S, S))
    edges = np.lib.format.open_memmap(out + '_edge.npy', 'w+', np.uint8, (n, S, S))
    scripts = [''] * n
    with Pool(int(os.environ.get('TN_PROCS', max(1, os.cpu_count() - 2))), initializer=init) as p:
        for k, (i, im, m, e, s) in enumerate(p.imap_unordered(work, range(n), chunksize=4)):
            imgs[i] = im; masks[i] = m; edges[i] = e; scripts[i] = s
            if k % 1000 == 0:
                print(k, flush=True); log.write(f'{k}\n'); log.flush()
    imgs.flush(); masks.flush(); edges.flush()
    json.dump(scripts, open(out + '_script.json', 'w'))
    log.write('done\n'); log.close()
    print('done', flush=True)
