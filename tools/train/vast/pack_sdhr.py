"""pack_sdhr.py OUT_DIR: the SD-HR job's extra bundle (on top of pack.py's flatscan_data.tar), made on Windows.

OUT_DIR/flatscan_sdhr.tar, unpacked into FS_DATA (/data):
  flat2/pages     page textures for synth.py: the orientation net's rendered pages (tools/orientnet/render_pages.py,
                  E:/fs_orient/pages/train) decorated by flat2/textures.py (pictures, bands, coupons, barcodes)
  flat2/anchor    real photos for learning without forgetting (flat2/make_anchor.py --copy, licence-clean sets only)
  models/highres_121x177.ckpt   SyntheticDoc-HR's released weights (MIT), the fine-tune's start
Run first:  python tools/train/flat2/textures.py E:/fs_orient/pages/train E:/fs_flat2/pages 6000
            python tools/train/flat2/make_anchor.py E:/fs_flat2/anchor <ft runs on Commons / WildReceipt / CORD> --copy"""
import os, sys, tarfile

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
with tarfile.open(os.path.join(out, 'flatscan_sdhr.tar'), 'w') as t:
    t.add('E:/fs_flat2/pages', arcname='flat2/pages')
    t.add('E:/fs_flat2/anchor', arcname='flat2/anchor')
    t.add(r'C:\Users\User\Desktop\model_zoo\SyntheticDoc\weights\highres_121x177.ckpt', arcname='models/highres_121x177.ckpt')
print('wrote', os.path.join(out, 'flatscan_sdhr.tar'))
