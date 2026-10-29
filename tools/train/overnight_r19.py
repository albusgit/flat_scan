"""overnight_r19.py: the round-19 overnight training, supervised (replaces overnight_r19.sh).

  python overnight_r19.py            (started detached, e.g. through WMI Win32_Process.Create, so that neither a
                                       Claude Code session ending nor its low-memory reap of background tasks stops it)

GPU stages, one after the other (each resumes from its newest checkpoint when restarted):
  1. page finder v2 (pagefind2/train.py): the synthetic pack so far -> phase A to PF_A steps; the pack rebuilt from
     everything the generator has made by then -> phase B to PF_STEPS
  2. SD-HR pass 3 (flat2/train_run2.py) from pass 2: synthetic (read live while the generator adds), real flat photos
     at a fixed share, UVDoc, anchors; held-out synthetic and held-out real flat photos evaluated every 1000 steps
Beside them on the CPU: the synthetic generator (flat2/make_both.py) at below-normal priority.

Memory (round 19: Claude Code killed jobs when RAM ran out):
  - every child process tree runs in its own Windows job object with a hard cap on committed memory (an allocation
    past it fails in that tree only) and is killed with the supervisor (no orphans holding the GPU or RAM);
  - a watchdog every 5 s: free RAM < SUSPEND_GB -> the generator is suspended (training goes on), back over
    RESUME_GB -> resumed; still < KILL_GB -> the generator is stopped and restarted later with one worker fewer
    (it skips finished jobs); C: free < DISK_GB -> the generator stops for good (training data is enough);
  - a stage that dies (out of memory or anything else) is restarted from its newest checkpoint once RAM allows, up
    to MAX_TRIES times. Every event goes to STATUS.md.
"""
import ctypes
import ctypes.wintypes as wt
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import time

PY = r'C:/Users/User/envs/flatscan_torch/Scripts/python.exe'
R = os.path.dirname(os.path.abspath(__file__))
W = 'C:/fs_train'
ST = os.path.join(W, 'STATUS.md')
PF_DIR, PF_OUT = os.path.join(R, 'pagefind2'), os.path.join(W, 'pf2', 'r19')
SD_DIR, SD_OUT = os.path.join(R, 'flat2'), 'E:/fs_train/sdhr3'
PF_A, PF_STEPS, SD_STEPS = 25000, 60000, 30000
PF_ARGS = ['--bs', '16', '--amp', 'bf16', '--channels_last', '1', '--compile', '0', '--fused', '1', '--tf32', '1',
           '--real_share', '0.6', '--eval', '2500']
VALS = [f'{W}/pf2/val_smartdoc.npz', f'{W}/pf2/val_doccorner.npz', f'{W}/pf2/val_synth.npz']
PF_SHARDS = ['E:/fs_flat2/both3/pf', f'{W}/both3/pf']
SD_SYNTH = f'{W}/both3/sdhr,E:/fs_flat2/both3/sdhr,{W}/real_sdhr@0.35'
GEN_WORKERS = 8          # 8 make as many photos/s as 12 (memory bandwidth bound) for 4 GB less RAM
CAP_GB = {'gen': 17, 'pf': 16, 'sd': 22, 'pack': 12}   # committed memory per job (measured peaks: generator 12.9, PF 10.1, SD-HR 15.7)
SUSPEND_GB, RESUME_GB, KILL_GB, DISK_GB = 3.0, 5.5, 1.8, 6.0
MAX_TRIES = 8

k32 = ctypes.WinDLL('kernel32', use_last_error=True)
ntdll = ctypes.WinDLL('ntdll')
k32.CreateJobObjectW.restype = wt.HANDLE
k32.OpenProcess.restype = wt.HANDLE
k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
k32.QueryInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p]
k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
k32.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
k32.CloseHandle.argtypes = [wt.HANDLE]
ntdll.NtSuspendProcess.argtypes = [wt.HANDLE]
ntdll.NtResumeProcess.argtypes = [wt.HANDLE]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wt.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wt.DWORD), ('Affinity', ctypes.c_size_t), ('PriorityClass', wt.DWORD),
                ('SchedulingClass', wt.DWORD)]


class EXT_LIMIT(ctypes.Structure):
    _fields_ = [('Basic', BASIC_LIMIT), ('Io', IO_COUNTERS), ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class MEMSTAT(ctypes.Structure):
    _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


JOB_MEMORY, KILL_ON_CLOSE, BREAKAWAY_OK, PRIORITY = 0x200, 0x2000, 0x800, 0x20
BELOW_NORMAL, CREATE_SUSPENDED, NO_WINDOW = 0x4000, 0x4, 0x08000000


def free_gb():
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def disk_gb(path='C:/'):
    return shutil.disk_usage(path).free / 2 ** 30


def note(msg):
    line = f'- {datetime.datetime.now():%H:%M} {msg}'
    print(line, flush=True)
    with open(ST, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


class Job:
    """a command in its own job object: memory cap, kill-on-close, suspend / resume / kill of the whole tree"""

    def __init__(self, name, cmd, cwd, log, cap_gb, below_normal=False):
        self.name, self.log_path = name, log
        self.h = k32.CreateJobObjectW(None, None)
        info = EXT_LIMIT()
        info.Basic.LimitFlags = JOB_MEMORY | KILL_ON_CLOSE | (PRIORITY if below_normal else 0)
        info.Basic.PriorityClass = BELOW_NORMAL if below_normal else 0
        info.JobMemoryLimit = int(cap_gb * 2 ** 30)
        if not k32.SetInformationJobObject(self.h, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), 'SetInformationJobObject')
        self.logf = open(log, 'a', encoding='utf-8')
        self.logf.write(f'\n==== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {" ".join(cmd)}\n')
        self.logf.flush()
        # started suspended, put in the job, then let go: no child can start outside the job
        self.p = subprocess.Popen(cmd, cwd=cwd, stdout=self.logf, stderr=subprocess.STDOUT, creationflags=CREATE_SUSPENDED | NO_WINDOW)
        if not k32.AssignProcessToJobObject(self.h, int(self.p._handle)):
            err = ctypes.get_last_error()
            self.p.kill()
            raise OSError(err, 'AssignProcessToJobObject')
        ntdll.NtResumeProcess(int(self.p._handle))
        self.suspended, self.t0 = False, time.time()

    def pids(self):
        n = 4096

        class LIST(ctypes.Structure):
            _fields_ = [('Assigned', wt.DWORD), ('InList', wt.DWORD), ('Ids', ctypes.c_size_t * n)]
        lst = LIST()
        if not k32.QueryInformationJobObject(self.h, 3, ctypes.byref(lst), ctypes.sizeof(lst), None):
            return []
        return [lst.Ids[i] for i in range(lst.InList)]

    def peak_gb(self):
        info = EXT_LIMIT()
        k32.QueryInformationJobObject(self.h, 9, ctypes.byref(info), ctypes.sizeof(info), None)
        return info.PeakJobMemoryUsed / 2 ** 30

    def _each(self, fn):
        for pid in self.pids():
            h = k32.OpenProcess(0x0800, False, pid)   # PROCESS_SUSPEND_RESUME
            if h:
                fn(h)
                k32.CloseHandle(h)

    def suspend(self):
        if not self.suspended:
            self._each(ntdll.NtSuspendProcess)
            self.suspended = True

    def resume(self):
        if self.suspended:
            self._each(ntdll.NtResumeProcess)
            self.suspended = False

    def poll(self):
        return self.p.poll()

    def kill(self):
        self.resume()
        k32.TerminateJobObject(self.h, 1)
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            pass

    def close(self):
        if self.poll() is None:
            self.kill()
        k32.CloseHandle(self.h)
        self.logf.close()

    def tail(self, n=3):
        try:
            with open(self.log_path, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f.readlines()[-200:] if l.strip() and 'Warning' not in l and 'pow_by_natural' not in l]
            return ' | '.join(lines[-n:])[-400:]
        except OSError:
            return ''


class Generator:
    """make_both.py: restarted after it dies (it skips the jobs already done), with a worker fewer after a memory kill"""

    def __init__(self):
        self.job, self.workers, self.stopped_for_good, self.restart_after = None, GEN_WORKERS, False, 0
        self.kills = 0

    def cmd(self):
        return [PY, 'make_both.py', f'{W}/pages2', f'{W}/both3', '38000', '--workers', str(self.workers), '--per_job', '40',
                '--seed', '310000', '--pass1', '../../../app/src/main/assets/models/uvdoc.onnx']

    def start(self):
        self.job = Job('generator', self.cmd(), SD_DIR, f'{W}/both3.log', CAP_GB['gen'], below_normal=True)
        note(f'generator started: {self.workers} workers, below-normal priority, cap {CAP_GB["gen"]} GB')

    def tick(self, free):
        if self.stopped_for_good:
            return
        if self.job is None:
            if time.time() > self.restart_after and free > RESUME_GB + 2 and disk_gb() > DISK_GB + 2:
                self.start()
            return
        rc = self.job.poll()
        if rc is not None:
            done = rc == 0
            note(f'generator {"finished" if done else f"stopped (exit {rc}): {self.job.tail(2)}"}; peak commit {self.job.peak_gb():.1f} GB')
            self.job.close()
            self.job = None
            if done:
                self.stopped_for_good = True
            else:
                self.workers = max(4, self.workers - 1)
                self.restart_after = time.time() + 60
            return
        if disk_gb() < DISK_GB:
            note(f'C: has {disk_gb():.1f} GB free: generator stopped for good (the data made so far is kept)')
            self.job.close()
            self.job, self.stopped_for_good = None, True
            return
        if free < KILL_GB and self.job.suspended:
            note(f'RAM free {free:.1f} GB with the generator suspended: generator stopped, restarts with fewer workers')
            self.kills += 1
            self.job.close()
            self.job = None
            self.workers = max(4, self.workers - 1)
            self.restart_after = time.time() + 300
        elif free < SUSPEND_GB and not self.job.suspended:
            self.job.suspend()
            note(f'RAM free {free:.1f} GB: generator suspended')
        elif free > RESUME_GB and self.job.suspended:
            self.job.resume()
            note(f'RAM free {free:.1f} GB: generator resumed')

    def stop(self):
        if self.job is not None:
            self.job.close()
            self.job = None


GEN = Generator()


def wait(job, label):
    """run a stage's job to its end while the watchdog looks after the generator; returns the exit code"""
    last_report = time.time()
    while True:
        rc = job.poll()
        free = free_gb()
        GEN.tick(free)
        if rc is not None:
            return rc
        if time.time() - last_report > 1800:   # a pulse every 30 min
            last_report = time.time()
            note(f'{label}: running {(time.time() - job.t0) / 60:.0f} min; {job.tail(1)[:200]}; RAM free {free:.1f} GB, '
                 f'C: {disk_gb():.0f} GB; generator {"suspended" if GEN.job and GEN.job.suspended else "on" if GEN.job else "off"}')
        time.sleep(5)


def run_stage(label, make_cmd, cwd, log, cap, done):
    """make_cmd(attempt) -> argv (resuming from the newest checkpoint); done() -> True when the stage's goal is met"""
    for attempt in range(1, MAX_TRIES + 1):
        if done():
            return True
        while free_gb() < 6:
            GEN.tick(free_gb())
            time.sleep(10)
        job = Job(label, make_cmd(attempt), cwd, log, cap)
        rc = wait(job, label)
        peak = job.peak_gb()
        tail = job.tail(3)
        job.close()
        if rc == 0 and done():
            note(f'{label}: done (peak commit {peak:.1f} GB)')
            return True
        note(f'{label}: exit {rc} on attempt {attempt} (peak commit {peak:.1f} GB): {tail}')
        time.sleep(30)
    note(f'{label}: gave up after {MAX_TRIES} attempts')
    return False


def pf_step():
    try:
        return json.load(open(os.path.join(PF_OUT, 'last.json')))['step']
    except (OSError, ValueError, KeyError):
        return 0


def sd_newest():
    """(step, path) of SD-HR's newest full checkpoint: last.pt (last.json) or step*.pt"""
    best = (0, '')
    for p in glob.glob(os.path.join(SD_OUT, 'step*.pt')):
        try:
            best = max(best, (int(os.path.basename(p)[4:10]), p))
        except ValueError:
            pass
    try:
        s = json.load(open(os.path.join(SD_OUT, 'last.json')))['step']
        if s > best[0] and os.path.exists(os.path.join(SD_OUT, 'last.pt')):
            best = (s, os.path.join(SD_OUT, 'last.pt'))
    except (OSError, ValueError, KeyError):
        pass
    return best


def build_pack(name):
    out = f'{W}/pf2/{name}.npz'
    log = f'{W}/pf2/pack_{name}.log'
    ok = run_stage(f'pack {name}', lambda _: [PY, 'data.py', 'synth', out] + PF_SHARDS, PF_DIR, log, CAP_GB['pack'],
                   lambda: os.path.exists(out) and os.path.getmtime(out) > T_START and 'saved' in open(log).read().split('====')[-1])
    ok = ok and run_stage(f'gutter {name}', lambda _: [PY, 'gutter.py', out], PF_DIR, log, CAP_GB['pack'],
                          lambda: 'gutter mean' in open(log).read().split('====')[-1])
    if ok:
        saved = [l for l in open(log).read().splitlines() if l.startswith('saved')]
        note(f'pack {name}: {saved[-1][:160] if saved else "?"}')
    return out if ok else None


def main():
    global T_START
    T_START = time.time()
    os.makedirs(PF_OUT, exist_ok=True)
    os.makedirs(SD_OUT, exist_ok=True)
    with open(ST, 'a', encoding='utf-8') as f:
        f.write(f'\n# Round 19 overnight run ({datetime.datetime.now():%Y-%m-%d %H:%M})\n')
    note(f'start: page finder {PF_STEPS} steps (pack rebuilt at {PF_A}), SD-HR {SD_STEPS} steps; RAM free {free_gb():.1f} GB, '
         f'C: {disk_gb():.0f} GB free; caps {CAP_GB} GB; supervisor pid {os.getpid()}')
    GEN.start()

    # 1. page finder
    pack = build_pack('synth_b') or f'{W}/pf2/synth_a.npz'
    pf_cmd = lambda synth, stop: (lambda _: [PY, 'train.py', '--synth', synth, '--real', f'{W}/pf2/train_real.npz', '--val'] + VALS +
                                  ['--out', PF_OUT, '--steps', str(PF_STEPS)] + PF_ARGS + (['--stop_at', str(stop)] if stop else []) +
                                  (['--resume', os.path.join(PF_OUT, 'last.pt')] if pf_step() else []))
    run_stage('page finder A', pf_cmd(pack, PF_A), PF_DIR, f'{PF_OUT}/train.log', CAP_GB['pf'], lambda: pf_step() >= PF_A)
    pack = build_pack('synth_c') or pack
    run_stage('page finder B', pf_cmd(pack, 0), PF_DIR, f'{PF_OUT}/train.log', CAP_GB['pf'], lambda: pf_step() >= PF_STEPS)
    try:
        ev = [json.loads(l) for l in open(os.path.join(PF_OUT, 'log.jsonl')) if '"score"' in l]
        b = max(ev, key=lambda r: r['score'])
        note(f'page finder: best mean IoU {b["score"]} at step {b["step"]}: ' +
             ', '.join(f'{k} {v["miou"]} ({v["fail"]}/{v["n"]} < 0.9)' for k, v in b['eval'].items()))
    except (OSError, ValueError, KeyError):
        pass

    # 2. SD-HR pass 3
    def sd_cmd(_):
        step, path = sd_newest()
        return ([PY, 'train_run2.py', '--synth', SD_SYNTH, '--held', 'E:/fs_flat2/held2', '--held_real', f'{W}/real_sdhr_val',
                 '--uvdoc', 'E:/fs_datasets/uvdoc/prep', '--anchor', 'E:/fs_flat2/anchor2', '--init', 'E:/fs_flat2/run2b/step015000.pt',
                 '--out', SD_OUT, '--counts', '10,3,3', '--micro', '4', '--steps', str(SD_STEPS), '--lr', '1e-5', '--enc_lr', '3e-6',
                 '--warmup', '300', '--eval', '1000', '--save', '0', '--ckpt', '500', '--real_sets', '', '--workers', '5',
                 '--live_reload', '300'] + (['--resume', path] if step else []))
    run_stage('SD-HR', sd_cmd, SD_DIR, f'{SD_OUT}/train.log', CAP_GB['sd'], lambda: sd_newest()[0] >= SD_STEPS)
    try:
        ev = [json.loads(l) for l in open(os.path.join(SD_OUT, 'log.jsonl')) if '"hr_flat"' in l]
        first, b = ev[0], min(ev, key=lambda r: r['hr_flat'])
        note(f'SD-HR: held-out real flat grid L1 x1000 {first["hr_flat"]:.2f} at the start -> best {b["hr_flat"]:.2f} at step {b["step"]}'
             f' (held synthetic at that step: ' + ', '.join(f'{k[4:]} {v:.1f}' for k, v in b.items() if k.startswith('syn_')) + ')')
    except (OSError, ValueError, KeyError, IndexError):
        pass

    # the generator may still run: let it finish its jobs (it is data for later passes), then stop
    while GEN.job is not None and not GEN.stopped_for_good:
        GEN.tick(free_gb())
        time.sleep(10)
    GEN.stop()
    note('all done')


T_START = 0
if __name__ == '__main__':
    try:
        main()
    except Exception as e:   # the status file says so; the job objects' kill-on-close stops the children
        note(f'supervisor error: {type(e).__name__}: {e}')
        raise
