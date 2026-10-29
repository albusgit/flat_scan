"""supervise_r19.py: the round-19 fine-tune of the rule + text-line net (overnight_r19.sh's training step), supervised
as tools/train/overnight_r19.py runs its stages: started detached (WMI Win32_Process.Create), so that neither a
Claude Code session ending nor its low-memory reap stops it; the trainer in its own job object with a cap on committed
memory; a watchdog that stops it when free RAM stays under KILL_GB and starts it again with a data worker fewer. The
pools are overnight_r19.sh's (made first). Every event goes to E:/fs_datasets/textnet/STATUS_r19.md.

  python supervise_r19.py"""
import ctypes
import datetime
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'train'))
from overnight_r19 import Job, free_gb   # noqa: E402

PY = r'C:/Users/User/envs/flatscan_torch/Scripts/python.exe'
ROOT = 'E:/fs_datasets/textnet'
OLD = 'C:/Users/User/fs_textnet/pools'
POOL = f'{ROOT}/pools_r19'
RUN = f'{ROOT}/run_r19'
ST = f'{ROOT}/STATUS_r19.md'
CAP_GB = 20              # the trainer and its data workers, committed memory (14 was hit at start: CUDA + 5 workers)
KILL_GB = 1.2            # free RAM under this for 30 s: stopped, started again with a worker fewer
WORKERS, MIN_WORKERS, MAX_TRIES = 4, 2, 6


def note(msg):
    line = f'- {datetime.datetime.now():%Y-%m-%d %H:%M} {msg}'
    print(line, flush=True)
    with open(ST, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def warm(prefix):
    """the product pool is on E: (a hard disk, C: is full): read once in order, so the trainer's random reads find it in
    the file cache (9.6 GB; 13 ms a random read from the disk)"""
    t = time.time()
    for s in ('_img', '_mask', '_edge'):
        with open(prefix + s + '.npy', 'rb') as f:
            while f.read(64 << 20):
                pass
    note(f'{prefix} read into the file cache in {time.time() - t:.0f} s; RAM free {free_gb():.1f} GB')


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # no sleep while this runs
    os.makedirs(RUN, exist_ok=True)
    os.environ.update(TN_INIT=f'{ROOT}/run1/best.pt', TN_HOURS='4', TN_LR='6e-4', TN_VAL_PROD=f'{POOL}/vprod')
    cmd = [PY, '-I', os.path.join(HERE, 'train_text.py'), RUN, f'{OLD}/text1,{POOL}/prod1', f'{OLD}/rule1', f'{OLD}/vtext', f'{OLD}/vrule']
    workers = WORKERS
    note(f'start: fine-tune from run1/best.pt, 4 h, lr 6e-4, {workers} data workers, cap {CAP_GB} GB; the data workers read '
         'the pools with file reads, not memory maps (round 19 attempts 1-6 were stopped by the watchdog: the maps pages)')
    warm(f'{POOL}/prod1')
    for attempt in range(1, MAX_TRIES + 1):
        while free_gb() < 4:
            time.sleep(10)
        os.environ['TN_WORKERS'] = str(workers)
        job = Job('textnet', cmd, HERE, f'{RUN}/train.txt', CAP_GB)
        low_since, last_pulse, killed = None, time.time(), False
        while job.poll() is None:
            free = free_gb()
            if free < KILL_GB:
                low_since = low_since or time.time()
                if time.time() - low_since > 30:
                    note(f'attempt {attempt}: free RAM {free:.1f} GB for 30 s: stopped')
                    job.kill()
                    killed = True
                    break
            else:
                low_since = None
            if time.time() - last_pulse > 1800:
                last_pulse = time.time()
                note(f'running {(time.time() - job.t0) / 60:.0f} min; {job.tail(1)[:200]}; RAM free {free:.1f} GB')
            time.sleep(5)
        rc, peak, tail = job.poll(), job.peak_gb(), job.tail(3)
        job.close()
        if rc == 0 and not killed:
            open(f'{RUN}/finished', 'w').close()
            note(f'done (peak commit {peak:.1f} GB): {tail}')
            return
        note(f'attempt {attempt}: exit {rc} (peak commit {peak:.1f} GB): {tail}')
        workers = max(MIN_WORKERS, workers - 1)
        time.sleep(30)
    note(f'gave up after {MAX_TRIES} attempts')


if __name__ == '__main__':
    main()
