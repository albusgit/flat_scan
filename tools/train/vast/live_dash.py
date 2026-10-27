"""live_dash.py PORT TOKEN ('-': the page at /, for the vast.ai portal's Tensorboard slot) [--rate R --budget B --steps S --target N]: a live progress page for run_sdhr2.sh on the box
itself (standard library only). http://<box>:<mapped port>/<TOKEN>/ is the page, which reloads /<TOKEN>/status.json
every 5 s; anything else is 404 (the address is public: the token keeps it out of casual reach). Reads only:
the upload sizes, $WORK/sdhr2 (data, checks, run/log.jsonl, export), nvidia-smi, /proc."""
import argparse
import http.server
import json
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument('port', type=int)
ap.add_argument('token')
ap.add_argument('--rate', type=float, default=1.116)
ap.add_argument('--budget', type=float, default=3.71)
ap.add_argument('--steps', type=int, default=8000)
ap.add_argument('--target', type=int, default=20000)
ap.add_argument('--run', default='run', help='the run folder under $WORK/sdhr2')
ap.add_argument('--synth', default='synth', help='synthetic folders under $WORK/sdhr2 (comma separated)')
ap.add_argument('--uploads', default='flatscan_code.tar:16,A_pages.tar:366,B_rest.tar:338', help='name:MB of each upload')
A = ap.parse_args()
WORK = os.environ.get('WORK', '/work')
S = os.path.join(WORK, 'sdhr2')
UPLOADS = [(n, int(mb) * 2 ** 20) for n, mb in (u.split(':') for u in A.uploads.split(','))]
_slow = {'t': 0, 'uv_mb': 0}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def boot_time():
    # the container's start: process 1's start in seconds since boot, plus the boot time
    try:
        hz = os.sysconf('SC_CLK_TCK')
        start = int(open('/proc/1/stat').read().rsplit(')', 1)[1].split()[19]) / hz
        btime = next(int(l.split()[1]) for l in open('/proc/stat') if l.startswith('btime'))
        return btime + start
    except Exception:
        return time.time()


START = boot_time()


def lines(path):
    try:
        with open(path, 'rb') as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def status():
    now = time.time()
    if now - _slow['t'] > 30:   # du over the UVDoc download is slow: every 30 s
        _slow['uv_mb'] = int((sh(f'du -sm {WORK}/uvdoc 2>/dev/null').split() or ['0'])[0])
        _slow['t'] = now
    have = sum(min(os.path.getsize(f'/root/{n}') if os.path.exists(f'/root/{n}') else 0, b) for n, b in UPLOADS)
    total = sum(b for _, b in UPLOADS)
    b_done = os.path.exists('/root/B_done')
    synth = sum(lines(f'{S}/{d}/index.jsonl') for d in A.synth.split(','))
    shards = sum(len(os.listdir(f'{S}/{d}/manifests')) for d in A.synth.split(',') if os.path.isdir(f'{S}/{d}/manifests'))
    uv_ok = os.path.exists(f'{WORK}/uvdoc/prep/index.json')
    log = []
    try:
        for l in open(f'{S}/{A.run}/log.jsonl'):
            try:
                log.append(json.loads(l))
            except ValueError:
                pass
    except OSError:
        pass
    train = [x for x in log if 'loss' in x]
    evals = [x for x in log if any(k.startswith('syn_') for k in x)]
    step = max((x['step'] for x in log), default=0)
    sp = train[-1]['sec_per_step'] if train else None
    boxlog = open('/root/box.log').read()[-4000:] if os.path.exists('/root/box.log') else ''
    pre = open(f'{S}/preflight.txt').read() if os.path.exists(f'{S}/preflight.txt') else ''
    chk = open(f'{S}/check.txt').read() if os.path.exists(f'{S}/check.txt') else ''
    exp = len([f for f in os.listdir(f'{S}/export') if f.startswith(A.run + '_')]) if os.path.isdir(f'{S}/export') else 0
    done = 'BOX_DONE' in boxlog
    problem = next((l for l in boxlog.splitlines()[::-1] if any(w in l for w in ('Traceback', 'Error', 'rror:', 'Killed', 'do not line up'))), '')
    gpu = (sh('nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits') + ',,').split(',')[:3]
    mem = sh("free -g | awk '/Mem:/{print $3, $2}'").split() + ['0', '0']

    def st(name, state, detail, eta=''):
        return {'stage': name, 'state': state, 'detail': detail, 'eta': eta}
    stages = [
        st('Upload bundles', 'done' if b_done else 'running', f'{have / 2**20:,.0f} of {total / 2**20:,.0f} MB on the box'),
        st('Set up the box', 'done' if shards or step else ('running' if boxlog else 'waiting'), 'Python packages'),
        st('Make training data', 'done' if synth >= A.target or step else ('running' if shards or 'samples there' in boxlog else 'waiting'),
           f'{synth:,} of {A.target:,} synthetic samples'),
        st('Fetch UVDoc shapes', 'done' if uv_ok else ('running' if _slow['uv_mb'] else 'waiting'), f'{_slow["uv_mb"]:,} MB downloaded'),
        st('Checks before training', 'done' if pre and step else ('running' if pre else 'waiting'),
           (chk.strip().splitlines() or ['precision and UVDoc alignment'])[-1][:160]),
        st('Train', 'done' if step >= A.steps else ('running' if step else 'waiting'), f'step {step:,} of {A.steps:,}' + (f', {sp:.2f} s a step' if sp else ''),
           f'{(A.steps - step) * sp / 60:.0f} min left' if sp and step < A.steps else ''),
        st('Export models', 'done' if done else ('running' if step >= A.steps else 'waiting'), f'{exp} ONNX files'),
    ]
    if problem and not done:
        stages.append(st('Problem in the log', 'failed', problem[-220:]))
    hours = (now - START) / 3600
    return {
        'now': now, 'hours': hours, 'cost': hours * A.rate, 'budget': A.budget, 'rate': A.rate,
        'gpu': {'util': gpu[0].strip(), 'mem_mb': gpu[1].strip(), 'watts': gpu[2].strip()},
        'cpu_load': open('/proc/loadavg').read().split()[0], 'cores': os.cpu_count(),
        'ram': {'used': mem[0], 'total': mem[1]},
        'stages': stages,
        'train': [{'step': x['step'], 'synth': x.get('synth_l1'), 'uvdoc': x.get('uvdoc_l1'), 'anchor': x.get('anchor_l1'),
                   'sps': x.get('sec_per_step')} for x in train],
        'evals': [{'step': x['step'], **{k[4:]: v for k, v in x.items() if k.startswith('syn_')}} for x in evals],
        'log_tail': boxlog[-1500:],
    }


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SD-HR run 2 (live)</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--fg:#1b1f24;--muted:#5d6570;--line:#dde1e6;--ok:#1f7a4d;--warn:#a86400;--bad:#b42318;--s1:#2f6fd6;--s2:#8a4fd1;--s3:#d1683b}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--panel:#1b1e23;--fg:#e8eaed;--muted:#9aa3ad;--line:#2c3138;--ok:#4cc38a;--warn:#e0a03a;--bad:#f0645a;--s1:#6ea3ff;--s2:#b78cff;--s3:#ff9a6a;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px max(16px,env(safe-area-inset-left)) 40px;display:grid;gap:16px}
h1{font:400 26px/1.2 Georgia,serif;margin:0}.take{color:var(--muted);margin:4px 0 0}
.tiles{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.lab{color:var(--muted);font-size:12px}.big{font:400 26px/1.2 Georgia,serif;font-variant-numeric:tabular-nums;margin-top:4px}
.sub{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;margin-top:4px}
.meter{height:8px;border-radius:4px;background:var(--line);margin-top:8px;overflow:hidden}.meter div{height:100%;background:var(--ok);width:0}
h2{font-size:15px;font-weight:600;margin:0 0 4px}.note{color:var(--muted);font-size:12px;margin:0 0 10px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-weight:500;font-size:12px}
td,th{padding:5px 10px 5px 0;border-bottom:1px solid var(--line);vertical-align:top}.num{text-align:right}
.pill{font-size:12px;padding:1px 8px;border-radius:999px;border:1px solid currentColor;white-space:nowrap}
.done{color:var(--ok)}.running{color:var(--warn)}.waiting{color:var(--muted)}.failed{color:var(--bad)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr))}.scroll{overflow-x:auto}
pre{white-space:pre-wrap;font-size:11px;color:var(--muted);max-height:220px;overflow:auto;margin:0}
.leg{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.leg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
</style></head><body><main>
<div><h1>SD-HR run 2</h1><p class="take" id="take">Loading…</p></div>
<div class="tiles">
<div class="card"><div class="lab">Spent so far</div><div class="big" id="cost">—</div><div class="sub" id="costsub"></div><div class="meter"><div id="bar"></div></div></div>
<div class="card"><div class="lab">Training</div><div class="big" id="step">—</div><div class="sub" id="eta"></div></div>
<div class="card"><div class="lab">Training data</div><div class="big" id="data">—</div><div class="sub" id="uv"></div></div>
<div class="card"><div class="lab">The box</div><div class="big" id="gpu">—</div><div class="sub" id="sys"></div></div>
</div>
<div class="card"><h2>Steps</h2><p class="note">Updates every 5 s. <span id="upd"></span></p><div class="scroll"><table><thead><tr><th>Step</th><th>State</th><th>Detail</th><th>Time left</th></tr></thead><tbody id="stages"></tbody></table></div></div>
<div class="grid">
<div class="card"><h2>Training error</h2><p class="note">Grid error (L1 × 1000, lower is better) by source, every 100 steps.</p><div id="chart"></div><div class="leg" id="leg"></div></div>
<div class="card"><h2>Held-out synthetic pages</h2><p class="note">Grid error (L1 × 1000) by shape at each checkpoint (every 2,000 steps).</p><div class="scroll" id="evals"><p class="note">The first checkpoint is at step 2,000.</p></div></div>
</div>
<div class="card"><h2>Log</h2><pre id="log"></pre></div>
</main><script>
const $=id=>document.getElementById(id),f2=v=>Number(v).toFixed(2),N=new Intl.NumberFormat('en');
const NAMES={done:'Done',running:'Running',waiting:'Waiting',failed:'Problem'};
function chart(rows){const el=$('chart');el.textContent='';const S=[['synth','Synthetic','--s1'],['uvdoc','UVDoc','--s2'],['anchor','Real photos (teacher)','--s3']].filter(([k])=>rows.some(r=>r[k]>0));
if(!rows.length){el.innerHTML='<p class="note">Training has not started yet.</p>';$('leg').textContent='';return}
const W=el.clientWidth||600,H=230,m={t:10,r:10,b:24,l:40},xs=rows.map(r=>r.step),x0=Math.min(...xs),x1=Math.max(...xs)||1,ym=Math.max(...rows.flatMap(r=>S.map(([k])=>(r[k]||0)*1000)))*1.08||1;
const X=v=>m.l+(v-x0)/Math.max(1,x1-x0)*(W-m.l-m.r),Y=v=>H-m.b-v/ym*(H-m.t-m.b);let s=`<svg width="100%" viewBox="0 0 ${W} ${H}">`;
for(let i=0;i<=4;i++){const v=ym*i/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${m.l-6}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--muted)">${v.toFixed(0)}</text>`}
for(const v of [x0,(x0+x1)/2,x1])s+=`<text x="${X(v)}" y="${H-6}" text-anchor="middle" font-size="11" fill="var(--muted)">${N.format(Math.round(v))}</text>`;
for(const [k,,c] of S)s+=`<polyline fill="none" stroke="var(${c})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${rows.filter(r=>r[k]!=null).map(r=>X(r.step)+','+Y(r[k]*1000)).join(' ')}"/>`;
el.innerHTML=s+'</svg>';$('leg').innerHTML=S.map(([k,n,c])=>{const last=rows[rows.length-1][k];return `<span><i style="background:var(${c})"></i>${n}${last!=null?': '+(last*1000).toFixed(1):''}</span>`}).join('')}
function evals(rows){if(!rows.length)return;const kinds=[...new Set(rows.flatMap(r=>Object.keys(r).filter(k=>k!=='step')))].sort();
let h='<table><thead><tr><th class="num">Step</th>'+kinds.map(k=>`<th class="num">${k}</th>`).join('')+'</tr></thead><tbody>';
for(const r of rows.slice().reverse())h+=`<tr><td class="num">${N.format(r.step)}</td>`+kinds.map(k=>`<td class="num">${r[k]!=null?Number(r[k]).toFixed(1):''}</td>`).join('')+'</tr>';$('evals').innerHTML=h+'</tbody></table>'}
async function tick(){try{const d=await (await fetch('status.json',{cache:'no-store'})).json();
$('cost').textContent='$'+f2(d.cost);$('costsub').textContent=`of $${f2(d.budget)} credit, ${f2(d.hours)} h at $${d.rate}/h (bandwidth not included)`;
const sh=Math.min(1,d.cost/d.budget);$('bar').style.width=(sh*100).toFixed(1)+'%';$('bar').style.background=sh>.85?'var(--bad)':sh>.6?'var(--warn)':'var(--ok)';
const by=Object.fromEntries(d.stages.map(s=>[s.stage,s]));$('step').textContent=by['Train'].detail.split(',')[0];$('eta').textContent=[by['Train'].detail.split(',')[1],by['Train'].eta].filter(Boolean).join(' · ');
$('data').textContent=by['Make training data'].detail;$('uv').textContent='UVDoc: '+by['Fetch UVDoc shapes'].detail;
$('gpu').textContent=`GPU ${d.gpu.util||0}%`;$('sys').textContent=`${(d.gpu.mem_mb/1024||0).toFixed(1)} GB VRAM, ${d.gpu.watts||0} W · CPU load ${d.cpu_load} of ${d.cores} cores · RAM ${d.ram.used} of ${d.ram.total} GB`;
$('stages').innerHTML='';for(const s of d.stages){const tr=document.createElement('tr');for(const [v,c] of [[s.stage],[NAMES[s.state]||s.state,'pill '+s.state],[s.detail],[s.eta]]){const td=document.createElement('td');if(c){const p=document.createElement('span');p.className=c;p.textContent=v;td.appendChild(p)}else td.textContent=v||'';tr.appendChild(td)}$('stages').appendChild(tr)}
const fail=d.stages.find(s=>s.state==='failed'),run=d.stages.filter(s=>s.state==='running').map(s=>s.stage.toLowerCase()),dn=d.stages.filter(s=>s.state==='done').length;
$('take').textContent=fail?'Stopped: '+fail.detail:dn===d.stages.length?'Finished: the models are ready to copy back.':(run.length?'Now: '+run.join(', ')+'. ':'')+`${dn} of ${d.stages.length} steps done.`;
chart(d.train);evals(d.evals);$('log').textContent=d.log_tail;$('upd').textContent='Last: '+new Date(d.now*1000).toLocaleTimeString();
}catch(e){$('upd').textContent='Box not answering ('+e.message+')'}}
tick();setInterval(tick,5000);addEventListener('resize',tick);
</script></body></html>'''


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split('?')[0]
        root = '' if A.token == '-' else f'/{A.token}'   # '-': at /, behind the vast.ai portal's login
        if p in (root or '/', f'{root}/'):
            body, ctype = PAGE.encode(), 'text/html; charset=utf-8'
        elif p == f'{root}/status.json':
            body, ctype = json.dumps(status()).encode(), 'application/json'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


http.server.ThreadingHTTPServer(('0.0.0.0', A.port), H).serve_forever()
