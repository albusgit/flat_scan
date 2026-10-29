"""pull.py USER/REPO OUT_DIR [--every 600] [--pf_eval CMD] [--sd_eval CMD]: while the box trains, its pushes (box.sh: runs/
pf3, runs/sd4, every 10 min) pulled to this PC, with the status, and each new checkpoint evaluated here (the app-level
checks need this PC's research sets, which never go to the box). CMD gets the checkpoint's path for {ckpt}, e.g.
  --pf_eval "python ../../pagefind2/compare.py --ckpt {ckpt} --out {ckpt}.cmp"
Stops with Ctrl+C; run it again any time (checkpoints already evaluated are marked {ckpt}.done)."""
import argparse
import glob
import json
import os
import subprocess
import time

from huggingface_hub import snapshot_download


def status(out):
    for d in sorted(glob.glob(os.path.join(out, 'runs', '*'))):
        log = os.path.join(d, 'log.jsonl')
        if os.path.exists(log):
            lines = open(log).read().strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                print(f'  {os.path.basename(d)}: ' + ', '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in last.items()), flush=True)
    box = os.path.join(out, 'runs', 'box.log')
    if os.path.exists(box):
        print('  box:', open(box).read().strip().splitlines()[-1], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('repo'); ap.add_argument('out'); ap.add_argument('--every', type=int, default=600)
    ap.add_argument('--pf_eval', default=''); ap.add_argument('--sd_eval', default='')
    a = ap.parse_args()
    while True:
        try:
            snapshot_download(a.repo, repo_type='dataset', allow_patterns=['runs/**'], local_dir=a.out)
        except Exception as e:   # the box may not have pushed yet; the network may blink
            print('pull failed:', str(e)[:200], flush=True)
        print(time.strftime('%H:%M'), flush=True)
        status(a.out)
        for run, cmd in (('pf3', a.pf_eval), ('sd4', a.sd_eval)):
            if not cmd:
                continue
            for ck in sorted(glob.glob(os.path.join(a.out, 'runs', run, 'step*.pt'))):
                if os.path.exists(ck + '.done'):
                    continue
                print('  eval', ck, flush=True)
                r = subprocess.run(cmd.format(ckpt=ck), shell=True)
                if r.returncode == 0:
                    open(ck + '.done', 'w').close()
        time.sleep(a.every)


if __name__ == '__main__':
    main()
