"""run_pf_r19b.py: page finder v2, second pass (round 19): from r19's best weights, with the fixes for what the
comparison with the app's finder showed (pagefind2/compare.py):
  - curved sides cut off: no paper / edge loss in a band along the real photos' 4-corner outlines (--quad_band; the
    synthetic pages' exact outlines teach the edge);
  - a page filling the photo broke up, photo-border leaks: zoomed-in samples (--zoom), pages filling / overfilling /
    cut by the frame;
  - paper found on photos with no document: licence-clean negatives (data.py negcrops: background strips of the
    training photos), 1 per batch.
Supervised like overnight_r19.py (job object with a memory cap, restarts from last.pt); start it detached (WMI).
STATUS.md gets the events; the trainer's log is OUT/train.log."""
import json
import os

import overnight_r19 as O

OUT = f'{O.W}/pf2/r19b'
STEPS = 20000
ARGS = ['--bs', '16', '--amp', 'bf16', '--channels_last', '1', '--compile', '0', '--fused', '1', '--tf32', '1',
        '--real_share', '0.45', '--neg', f'{O.W}/pf2/train_neg2.npz', '--neg_share', '0.0625', '--quad_band', '0.05',
        '--zoom', '0.3', '--init', f'{O.W}/pf2/r19/best.pt', '--lr', '6e-4', '--eval', '2500']


def step():
    try:
        return json.load(open(os.path.join(OUT, 'last.json')))['step']
    except (OSError, ValueError, KeyError):
        return 0


def cmd(_):
    return ([O.PY, 'train.py', '--synth', f'{O.W}/pf2/synth_c.npz', '--real', f'{O.W}/pf2/train_real.npz', '--val'] + O.VALS +
            ['--out', OUT, '--steps', str(STEPS)] + ARGS + (['--resume', os.path.join(OUT, 'last.pt')] if step() else []))


if __name__ == '__main__':
    O.GEN.stopped_for_good = True   # no synthetic generator here (run_stage's watchdog would start overnight_r19's)
    os.makedirs(OUT, exist_ok=True)
    O.note(f'page finder r19b: {STEPS} steps from r19/best.pt (quad band, zoom, negatives); supervisor pid {os.getpid()}')
    try:
        O.run_stage('page finder r19b', cmd, O.PF_DIR, f'{OUT}/train.log', O.CAP_GB['pf'], lambda: step() >= STEPS)
        ev = [json.loads(l) for l in open(os.path.join(OUT, 'log.jsonl')) if '"score"' in l]
        b = max(ev, key=lambda r: r['score'])
        O.note(f'page finder r19b: best mean IoU {b["score"]} at step {b["step"]}: ' +
               ', '.join(f'{k} {v["miou"]} ({v["fail"]}/{v["n"]} < 0.9)' for k, v in b['eval'].items()))
    except Exception as e:
        O.note(f'page finder r19b: supervisor error: {type(e).__name__}: {e}')
        raise
