"""make_data.py PAGES OUT N SEED: refinement pilot data, one worker (run several with different seeds / OUTs).

Per sample: a synthetic photo (tools/train/flat2/synth.py) with its exact page -> photo map G; the app's stage 1 on
it (pipeline_cli, nets on, line stage off, as the phone runs it) dumping the backward map B; the stage-1 page the
refiner sees (targets.render) and the composition targets (targets.build). Written to OUT:
  photo/<id>.jpg   the photo           map/<id>.npz   B (x, y) and G (2 x GH x GW, 321 points across)
  in/<id>.jpg      the stage-1 page    t/<id>.npz     d, valid, M, frame
  index.jsonl      id, kind, aspect, hw, stats (or lost: stage 1 kept under half the page in its frame)
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'train', 'flat2'))
sys.path.insert(0, HERE)
import geom  # noqa: E402
import synth  # noqa: E402
import targets  # noqa: E402

CLI = os.path.join(REPO, 'tools', 'desktop_check', 'build_new', 'pipeline_cli.exe')
OPENCV_BIN = 'C:/Users/User/Android/opencv/build/x64/vc16/bin'
GWD = 321


def stage1(photo_path, map_path):
    env = dict(os.environ, FS_NO_LINES='1', FS_RULE_NET='1', FS_NETS='1', FS_DUMP_MAP=map_path,
               FS_MODELS=os.path.join(REPO, 'app', 'src', 'main', 'assets', 'models'))
    env['PATH'] = OPENCV_BIN + os.pathsep + env.get('PATH', '')
    # forward slashes: the CLI writes the photo path into its JSON unescaped
    p = subprocess.run([CLI, photo_path.replace(os.sep, '/'), '-', 'net'], env=env, capture_output=True, text=True, timeout=300)
    try:
        return json.loads(p.stdout)
    except ValueError:
        return None


def main():
    pages_dir, out, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    for sub in ('photo', 'map', 'in', 't'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    pages = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(pages_dir) for f in fs if f.lower().endswith(('.jpg', '.png')))
    done = set()
    ip = os.path.join(out, 'index.jsonl')
    if os.path.exists(ip):   # resume after a stop
        done = {json.loads(l)['id'] for l in open(ip)}
    index = open(ip, 'a')
    tmp = tempfile.mkdtemp(prefix='fsref_')
    t0, made, i = time.time(), len(done), 0
    while made < n:
        sid = f'{seed}_{i:05d}'
        rng = np.random.default_rng((seed, i))
        i += 1
        if sid in done:
            continue
        sm = synth.sample(rng, pages)
        if sm is None:
            continue
        hw = sm['hw']
        GHd = int(round((GWD - 1) * hw)) + 1
        gs, gt = np.meshgrid(np.linspace(0, 1, GWD), np.linspace(0, 1, GHd))
        g2, _ = sm['cam'].project(geom.mesh_points(sm['P'], gs, gt).reshape(-1, 3))
        G = g2.reshape(GHd, GWD, 2).transpose(2, 0, 1).astype(np.float32)
        pp = os.path.join(out, 'photo', sid + '.jpg')
        cv2.imwrite(pp, sm['photo'], [cv2.IMWRITE_JPEG_QUALITY, 97])
        row = dict(id=sid, kind=sm['kind'], aspect=sm['aspect'], hw=hw)
        mp = os.path.join(tmp, sid + '.npz')
        js = stage1(pp, mp)
        if js is None or not js.get('ok') or not os.path.exists(mp):
            row['lost'] = 'stage1'
        else:
            with np.load(mp) as B:
                bx, by = B['x'], B['y']
            os.remove(mp)
            np.savez_compressed(os.path.join(out, 'map', sid + '.npz'), x=bx, y=by, G=G)
            r = targets.build(bx, by, G)
            if r is None:
                row['lost'] = 'page'
            else:
                cv2.imwrite(os.path.join(out, 'in', sid + '.jpg'), targets.render(sm['photo'], bx, by), [cv2.IMWRITE_JPEG_QUALITY, 92])
                np.savez_compressed(os.path.join(out, 't', sid + '.npz'), d=r['d'], valid=r['valid'], M=r['M'], frame=np.int32(r['frame']))
                row.update(stats=r['stats'], mode=js.get('mode'), page_hw=js.get('page_hw'))
        index.write(json.dumps(row) + '\n')
        index.flush()
        made += 1
        if made % 50 == 0:
            print(f'{made}/{n} {(time.time() - t0) / max(made - len(done), 1):.2f} s per sample', flush=True)
    print(f'done: {made} samples in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
