"""Bundle what the GPU box needs into two tar files (run on the Windows machine; the training data is made there).

  python pack.py OUT_DIR
OUT_DIR/flatscan_data.tar   ~1.2 GB, unpacks as the FS_DATA folder (/data):
    round4/seg, round4/assets/{gfonts,bsds,scans}   the page generator (shadow pages) and its fonts / photos
    filters_r11/shadow_gt/{jung2018,osr2020}, raw/rdd/test   real shadow pairs (validation only)
    raw/dir300_small, sets/dir300.json              dir300 shrunk for validation (photo long side 2000 px, scan
                                                    in grey at the DocUNet 598400 px), paths as the scripts expect
    textnet/udhr2, textnet/run1/best.pt             the corpus source and the shipped rule + text-line net
    models/uvdoc_best_model.pkl                     UVDoc's PyTorch weights (MIT), the flattening fine-tune's start
    round4/real_photos                              the 34 real photos (contact sheets)
OUT_DIR/flatscan_code.tar   the repo's tools/ and the app's current models (the scripts' --init defaults)"""
import json, os, shutil, sys, tarfile

import cv2
import numpy as np

SRC = 'E:/fs_datasets'
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
UVDOC_PKL = 'C:/Users/User/Desktop/model_zoo/UVDoc/model/best_model.pkl'


def main():
    out = sys.argv[1]
    stage = os.path.join(out, 'stage')
    os.makedirs(stage, exist_ok=True)
    copy = [('round4/seg', None), ('round4/assets/gfonts', None), ('round4/assets/bsds', None), ('round4/assets/scans', None),
            ('filters_r11/shadow_gt/jung2018', None), ('filters_r11/shadow_gt/osr2020', None), ('raw/rdd/test', None),
            ('textnet/udhr2', None), ('round4/real_photos', None)]
    for rel, _ in copy:
        d = os.path.join(stage, rel)
        if not os.path.exists(d):
            shutil.copytree(os.path.join(SRC, rel), d, ignore=shutil.ignore_patterns('*.f32', '__pycache__'))
    os.makedirs(os.path.join(stage, 'textnet', 'run1'), exist_ok=True)
    shutil.copy(os.path.join(SRC, 'textnet/run1/best.pt'), os.path.join(stage, 'textnet/run1/best.pt'))
    os.makedirs(os.path.join(stage, 'models'), exist_ok=True)
    shutil.copy(UVDOC_PKL, os.path.join(stage, 'models', 'uvdoc_best_model.pkl'))
    # dir300, shrunk
    s = json.load(open(os.path.join(SRC, 'sets', 'dir300.json')))
    for sub in ('dist', 'gt'):
        os.makedirs(os.path.join(stage, 'raw', 'dir300_small', sub), exist_ok=True)
    for it in s['items']:
        p = os.path.join(stage, 'raw', 'dir300_small', 'dist', it['id'] + '.jpg')
        g = os.path.join(stage, 'raw', 'dir300_small', 'gt', it['id'] + '.png')
        if not os.path.exists(p):
            a = cv2.imread(it['photo'])
            k = min(1.0, 2000 / max(a.shape[:2]))
            cv2.imwrite(p, cv2.resize(a, (round(a.shape[1] * k), round(a.shape[0] * k)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            b = cv2.imread(it['gt'], cv2.IMREAD_GRAYSCALE)
            k = np.sqrt(598400 / b.size)
            cv2.imwrite(g, cv2.resize(b, (round(b.shape[1] * k), round(b.shape[0] * k)), interpolation=cv2.INTER_AREA))
        it['photo'] = 'E:/fs_datasets/raw/dir300_small/dist/' + it['id'] + '.jpg'
        it['gt'] = 'E:/fs_datasets/raw/dir300_small/gt/' + it['id'] + '.png'
    os.makedirs(os.path.join(stage, 'sets'), exist_ok=True)
    json.dump(s, open(os.path.join(stage, 'sets', 'dir300.json'), 'w'), indent=1)
    with tarfile.open(os.path.join(out, 'flatscan_data.tar'), 'w') as t:
        for name in sorted(os.listdir(stage)):
            t.add(os.path.join(stage, name), arcname=name)
    with tarfile.open(os.path.join(out, 'flatscan_code.tar'), 'w') as t:
        t.add(os.path.join(REPO, 'tools'), arcname='flat_scan/tools', filter=lambda ti: None if '__pycache__' in ti.name or '/build' in ti.name.replace('\\', '/') else ti)
        for f in ('shadenet.onnx', 'rulenet.onnx', 'uvdoc.onnx'):
            t.add(os.path.join(REPO, 'app/src/main/assets/models', f), arcname='flat_scan/app/src/main/assets/models/' + f)
    for f in ('flatscan_data.tar', 'flatscan_code.tar'):
        print(f, round(os.path.getsize(os.path.join(out, f)) / 2 ** 30, 2), 'GB')


if __name__ == '__main__':
    main()
