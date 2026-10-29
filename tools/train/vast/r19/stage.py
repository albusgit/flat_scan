"""stage.py OUT_DIR [--parts code,weights,...]: the round-19 bundle for the rented box, as plain .tar parts (the data are
JPEGs and compressed .npz: zstd saved < 3 %), each one a file the box downloads in parallel (box.sh fetch) and
unpacks into its data folder. Built on the PC before renting, then uploaded to the store (upload.py), so the box's
first minutes are a fast download instead of an upload from home.

Parts (licence-clean or our own only: research sets never leave this PC):
  code     tools/ (no caches, no build folders). Not the app's models: the box needs none of them (SD-HR's real-set
           check, which uses uvdoc.onnx, runs on this PC: pull.py)
  weights  ImageNet MobileNetV3 (timm, Apache-2.0), page finder r19 best, SD-HR pass 3 (EMA weights only), the
           original SD-HR (anchor teacher). Not the round-4 page finder: the box trains with --no_teacher (book
           gutters from the generator's spreads), and that net's training data is not licence-clean
  pages    the page images the generators draw (pages2: our own renders)
  real     train_real_sam_clean.npz (SAM outlines, sam_pack.py, checked side by side and made whole: sam_clean.py) else
           train_real_sam.npz else train_real.npz, train_neg2.npz, real_sdhr (flat real
           photos for SD-HR), real_sdhr_val
  uvdoc    UVDoc's prepared grids (MIT; its 12 GB of photos the box fetches from the source itself)
  anchors  anchor2 (the original net as teacher on licence-clean photos)
  val      val_smartdoc, val_doccorner_clean (DocCorner val without research / Flickr sources), val_synth, held2
"""
import argparse
import os
import shutil
import tarfile
import time

import torch

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
W = 'C:/fs_train'


def parts():
    sam = next((f for f in (f'{W}/pf2/train_real_sam_clean.npz', f'{W}/pf2/train_real_sam.npz', f'{W}/pf2/train_real.npz') if os.path.exists(f)),
               f'{W}/pf2/train_real.npz')
    return {
        'code': [(f'{R}/tools', 'tools')],
        'weights': [('E:/fs_datasets/round4/assets/weights/mobilenetv3_large_100_ra-f55367f5.pth',
                     'round4/assets/weights/mobilenetv3_large_100_ra-f55367f5.pth'),
                    (f'{W}/pf2/r19/best.pt', 'weights/pf_r19_best.pt'),
                    ('@sdhr_ema', 'weights/sdhr_p3s30k.pt'),
                    (r'C:\Users\User\Desktop\model_zoo\SyntheticDoc\weights\highres_121x177.ckpt', 'models/highres_121x177.ckpt')],
        'pages': [(f'{W}/pages2', 'pages2')],
        'real': [(sam, 'real/train_real_sam.npz'),
                 (f'{W}/pf2/train_neg2.npz', 'real/train_neg2.npz'),
                 (f'{W}/real_sdhr', 'real/real_sdhr'), (f'{W}/real_sdhr_val', 'real/real_sdhr_val')],
        'uvdoc': [('E:/fs_datasets/uvdoc/prep', 'uvdoc/prep'), ('E:/fs_datasets/uvdoc/split.json', 'uvdoc/split.json')],
        'anchors': [('E:/fs_flat2/anchor2', 'anchors/anchor2')],
        'val': [(f'{W}/pf2/val_smartdoc.npz', 'val/val_smartdoc.npz'), (f'{W}/pf2/val_doccorner_clean.npz', 'val/val_doccorner_clean.npz'),
                (f'{W}/pf2/val_synth.npz', 'val/val_synth.npz'), ('E:/fs_flat2/held2', 'val/held2')],
    }


def skip(ti):
    n = ti.name
    if '__pycache__' in n or n.endswith('.pyc') or '/.git' in n or '/build/' in n or n.endswith('/build'):
        return None
    return ti


def sdhr_ema(tmp):
    """SD-HR pass 3's EMA weights only (the checkpoint also holds the optimizer: 131 MB -> ~35 MB)"""
    ck = torch.load('E:/fs_train/sdhr3/step030000.pt', map_location='cpu', weights_only=False)
    torch.save({'state_dict': ck['state_dict'], 'step': ck['step']}, tmp)
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--parts', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'box.sh'), os.path.join(a.out, 'box.sh'))   # the box's first download
    want = [p for p in a.parts.split(',') if p] or list(parts())
    for name in want:
        t0 = time.time()
        dst = os.path.join(a.out, name + '.tar')
        with tarfile.open(dst + '.tmp', 'w') as tf:
            for src, arc in parts()[name]:
                if src == '@sdhr_ema':
                    src = sdhr_ema(os.path.join(a.out, '_sdhr_ema.pt'))
                if not os.path.exists(src):
                    print(f'  MISSING {src}')
                    continue
                tf.add(src, arcname=arc, filter=skip)
        os.replace(dst + '.tmp', dst)
        print(f'{name}: {os.path.getsize(dst) / 2 ** 30:.2f} GB in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
