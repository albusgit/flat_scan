"""Round-4 behaviour of the app's pipeline vs the reference photo_pipeline.py on the real photos.

  python tools/round4_check.py APP_DIR REF_DIR [GT_JSON]

APP_DIR: pipeline_cli outputs (NAME.json, tools/desktop_check); REF_DIR: NAME/info.json from
`python code/photo_pipeline.py PHOTO REF_DIR/NAME surface` (with or without PAGE_MODEL).
Compares: completion (crash fixes), the quality gate (needs_manual_crop), the applied rotation
(no automatic 180), and the surface-fit residual."""
import json, os, sys

app_dir, ref_dir = sys.argv[1:3]
names = sorted(f[:-5] for f in os.listdir(app_dir) if f.endswith('.json') and not f.startswith('parity'))
ROT = {None: 'none', -1: 'none', 0: 'CW', 1: '180', 2: 'CCW'}
n_ok_app = n_ok_ref = gate_same = reasons_same = rot_same = 0
rows = []
for n in names:
    a = json.load(open(os.path.join(app_dir, n + '.json')))
    rp = os.path.join(ref_dir, n, 'info.json')
    r = json.load(open(rp)) if os.path.exists(rp) else None
    n_ok_app += bool(a['ok'])
    n_ok_ref += r is not None
    ga = a['needs_manual_crop'] if a['ok'] else ['(failed)']
    gr = r['needs_manual_crop'] if r else ['(failed)']
    kind = lambda g: sorted(x.split(' ')[0] + ' ' + x.split(' ')[1] for x in g)   # reason kinds without numbers
    gate_same += bool(ga) == bool(gr)
    reasons_same += kind(ga) == kind(gr)
    ra, rr = ROT[a.get('rot')], ROT[r.get('rotated') if r else None]
    rot_same += ra == rr
    print(f"{n:36s} app {'ok ' if a['ok'] else 'ERR'} gate {'; '.join(ga) or '-':42s} | ref {'ok ' if r else 'ERR'} gate {'; '.join(gr) or '-':42s} "
          f"| fit px app {a.get('fit_median_full_px', -1):7.2f} ref {r.get('fit_median_px', float('nan')) if r else float('nan'):7.2f} "
          f"| rot app {ra:4s} ref {rr:4s} | app mode {a['mode']}")
N = len(names)
print(f"\ncompleted: app {n_ok_app}/{N}, reference {n_ok_ref}/{N}")
print(f"quality gate flags the same photos: {gate_same}/{N}; same reason kinds: {reasons_same}/{N}")
print(f"same applied rotation: {rot_same}/{N}; app 180-degree turns: {sum(1 for n in names if json.load(open(os.path.join(app_dir, n + '.json'))).get('rot') == 1)}")
