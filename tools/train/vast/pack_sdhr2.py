"""pack_sdhr2.py OUT_DIR: everything run_sdhr2.sh needs, made on Windows (round 18).

OUT_DIR/flatscan_sdhr2.tar, unpacked into FS_DATA (/data):
  flat2/pages2       page textures (flat2/textures.py --forms 0.3 from the orientation net's train pages)
  flat2/held2        400 held-out synthetic samples (make_data.py on textures from the held pages: no shared page)
  flat2/anchor2      curated anchor photos (flat2/anchor_curate.py; licence-clean sets only), relative paths
  models/highres_121x177.ckpt   SyntheticDoc-HR's released weights (MIT): the start and the teacher
  raw/dir300_small, sets/dir300.json   dir300 shrunk (pack.py's stage): the net-only real check (not trained on)
OUT_DIR/flatscan_code.tar   the repo's tools/ and the app's uvdoc.onnx (pass 1), unpacked into /root"""
import os
import sys
import tarfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
STAGE = 'E:/fs_datasets/vast_pack/stage'
CKPT = r'C:\Users\User\Desktop\model_zoo\SyntheticDoc\weights\highres_121x177.ckpt'


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    skip = lambda ti: None if '__pycache__' in ti.name or ti.name.endswith('.tmp') else ti
    with tarfile.open(os.path.join(out, 'flatscan_sdhr2.tar'), 'w') as t:
        for src, arc in (('E:/fs_flat2/pages2', 'flat2/pages2'), ('E:/fs_flat2/held2', 'flat2/held2'),
                         ('E:/fs_flat2/anchor2', 'flat2/anchor2'), (CKPT, 'models/highres_121x177.ckpt'),
                         (STAGE + '/raw/dir300_small', 'raw/dir300_small'), (STAGE + '/sets/dir300.json', 'sets/dir300.json')):
            t.add(src, arcname=arc, filter=skip)
    with tarfile.open(os.path.join(out, 'flatscan_code.tar'), 'w') as t:
        t.add(os.path.join(REPO, 'tools'), arcname='flat_scan/tools',
              filter=lambda ti: None if '__pycache__' in ti.name or '/build' in ti.name.replace('\\', '/') else ti)
        t.add(os.path.join(REPO, 'app/src/main/assets/models/uvdoc.onnx'), arcname='flat_scan/app/src/main/assets/models/uvdoc.onnx')
    for f in ('flatscan_sdhr2.tar', 'flatscan_code.tar'):
        print(f, round(os.path.getsize(os.path.join(out, f)) / 2 ** 30, 2), 'GB')


if __name__ == '__main__':
    main()
