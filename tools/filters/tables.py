"""Markdown tables from eval_filters.py score files.
  python -I tables.py SET LABEL=RUN:FILTER [...] [--metrics a,b,c]"""
import json, os, sys

O = os.environ.get('FS_FILTER_OUT', r'C:\Users\User\Desktop\flatscan_out\r11\filters')
args = [a for a in sys.argv[2:] if not a.startswith('--') and '=' in a]
mi = sys.argv.index('--metrics') if '--metrics' in sys.argv else -1
metrics = sys.argv[mi + 1].split(',') if mi > 0 else ['marks_lost_sum', 'paper_white', 'paper_white_p10', 'paper_unev', 'ink_L', 'sat_med',
                                                       'sat_worst_p10', 'white_regions_sum', 'halo', 'dE', 'psnr_fit', 'ssim_fit', 'bg_resid', 'bg_chroma']
rows = []
for a in args:
    lab, spec = a.split('=', 1)
    run, k = spec.rsplit(':', 1)
    f = os.path.join(O, 'runs', run, sys.argv[1], 'scores.json')
    s = json.load(open(f))['summary'].get(k, {})
    rows.append((lab, s))
ms = [m for m in metrics if any(m in s for _, s in rows)]
print('| ' + sys.argv[1] + ' | ' + ' | '.join(ms) + ' |')
print('|---' * (len(ms) + 1) + '|')
for lab, s in rows:
    print('| ' + lab + ' | ' + ' | '.join(str(s.get(m, '')) for m in ms) + ' |')
