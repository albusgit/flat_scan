"""dash.py: a live page for the round-19 box (box.sh's runs): progress, losses, evaluations, GPUs, time and cost.

    python dash.py [--work /root/fs/work] [--port 17070] [--price 1.2]

Standard library only. Serves on 127.0.0.1 (vast.ai's portal / an SSH tunnel put it in front of the browser). / is the
page, /api the numbers it polls every 5 s."""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

A = None


def jsonl(path, keep=4000):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('{'):
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out[-keep:]


def tail(path, n=60):
    try:
        with open(path, errors='replace') as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def gpus():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        rows = []
        for line in q.strip().splitlines():
            i, name, u, mu, mt, t, p = [x.strip() for x in line.split(',')]
            rows.append(dict(i=int(i), name=name, util=float(u), mem=float(mu) / 1024, mem_total=float(mt) / 1024, temp=float(t),
                             power=float(p) if p not in ('[N/A]', '') else None))
        return rows
    except Exception as e:  # noqa: BLE001
        return [dict(error=str(e))]


def mem():
    d = {}
    try:
        for line in open('/proc/meminfo'):
            k, v = line.split(':')
            d[k] = int(v.split()[0]) / 1024 / 1024
        return dict(total=d['MemTotal'], used=d['MemTotal'] - d['MemAvailable'])
    except (OSError, KeyError):
        return {}


def job(name, steps):
    recs = jsonl(os.path.join(A.work, name, 'log.jsonl'))
    train = [r for r in recs if 'loss' in r]
    evals = [r for r in recs if 'loss' not in r and r.get('step') is not None]
    step = max([r['step'] for r in recs], default=0)
    rate = None
    t = [r for r in train if 'sec' in r]
    if len(t) >= 2:
        a, b = t[max(0, len(t) - 6)], t[-1]
        if b['sec'] > a['sec']:
            rate = (b['step'] - a['step']) / (b['sec'] - a['sec'])
    ckpts = sorted(f for f in os.listdir(os.path.join(A.work, name)) if f.startswith('step') and f.endswith('.pt')) \
        if os.path.isdir(os.path.join(A.work, name)) else []
    pushed = [f for f in ckpts if os.path.exists(os.path.join(A.work, name, f + '.pushed'))]
    try:
        idle = time.time() - os.path.getmtime(os.path.join(A.work, name, 'log.jsonl'))
    except OSError:
        idle = 0
    stopped = bool(recs) and idle > 600 and step < steps
    if stopped:
        rate = None
    return dict(step=step, steps=steps, rate=rate, stopped=stopped, eta_s=(steps - step) / rate if rate else None, train=train[-1500:],
                evals=evals, ckpts=len(ckpts), pushed=len(pushed), log=tail(os.path.join(A.work, name, 'train.log'), 12))


PARTS = ['code', 'weights', 'pages', 'real', 'uvdoc', 'anchors', 'val']
_UV = []   # (time, bytes) samples of the UVDoc photo folder: its download speed


def dir_bytes(path):
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


def downloads():
    """the bundle parts (box.sh's 'got X' / 'have X' / 'FAILED X' lines, the .tar landing in work/dl) and the UVDoc
    photos (fetch_zip.py's 'n/N x GB' lines, the folder's size over time)"""
    state = {}
    for line in tail(os.path.join(A.work, 'box.log'), 400):
        m = re.match(r'\S+ (got|have|FAILED) (\w+)$', line.strip())
        if m and m.group(2) in PARTS:
            state[m.group(2)] = 'failed' if m.group(1) == 'FAILED' else 'done'
    dl = os.path.join(A.work, 'dl')
    for part in PARTS:
        if part not in state and os.path.exists(os.path.join(dl, part + '.tar')):
            state[part] = f'downloading ({os.path.getsize(os.path.join(dl, part + ".tar")) / 1e9:.2f} GB)'
    total = None
    for line in tail(os.path.join(A.work, 'uvdoc.log'), 400):
        m = re.match(r'img/: (\d+) members, ([\d.]+) GB', line.strip())
        if m:
            total = (int(m.group(1)), float(m.group(2)))
    img = os.path.join(os.path.dirname(A.work), 'data', 'uvdoc', 'img')
    n = len(os.listdir(img)) if os.path.isdir(img) else 0
    b = dir_bytes(img)
    now = time.time()
    _UV.append((now, b))
    while len(_UV) > 2 and now - _UV[0][0] > 120:
        _UV.pop(0)
    rate = (b - _UV[0][1]) / (now - _UV[0][0]) if now - _UV[0][0] > 5 else None
    alive = False
    try:
        alive = os.path.exists(f'/proc/{int(open(os.path.join(A.work, "uvdoc.pid")).read())}')
    except (OSError, ValueError):
        pass
    return dict(parts=[[k, state.get(k, 'waiting')] for k in PARTS],
                uvdoc=dict(files=n, gb=b / 1e9, total_files=total[0] if total else 20000, total_gb=total[1] if total else 10.73,
                           rate=rate, running=alive))


def api():
    box = os.path.join(A.work, 'box.log')
    try:
        started = os.path.getctime(os.path.join(A.work, 'pip.log'))
    except OSError:
        started = None
    du = shutil.disk_usage('/')
    return dict(now=time.time(), started=started, price=A.price, gpus=gpus(), ram=mem(),
                disk=dict(used=du.used / 1e9, total=du.total / 1e9), box=tail(box, 40),
                dl=downloads(), pf=job('pf3', int(os.environ.get('PF_STEPS', 30000))), pfb=job('pf3b', 30000), sd=job('sd5', 20000) if os.path.isdir(os.path.join(A.work, 'sd5')) else job('sd4', int(os.environ.get('SD_STEPS', 30000))))


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FlatScan Training</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--mute:#667085;--line:#e4e7ec;--a:#2f6fe4;--b:#e4572e;--c:#17a673;--d:#9b51e0}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e6e9ef;--mute:#98a2b3;--line:#2b3240}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}
.sub{color:var(--mute);margin-bottom:16px}.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;min-width:0}
.big{font-size:26px;font-weight:600}.mute{color:var(--mute)}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:8px 0}
.bar>i{display:block;height:100%;background:var(--a)}pre{white-space:pre-wrap;word-break:break-word;font:12px/1.4 ui-monospace,Consolas,monospace;margin:0;max-height:320px;overflow:auto}
.two{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));margin-top:12px}canvas{width:100%!important;height:240px!important}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
</style></head><body><main>
<h1>FlatScan round 19 training</h1><div class="sub" id="sub">loading…</div>
<div class="grid">
 <div class="card"><h2>Page finder v3 (lr 6e-4, GPU 0)</h2><div class="big" id="pfp">–</div><div class="bar"><i id="pfb" style="width:0"></i></div><div class="mute" id="pfm"></div></div>
 <div class="card"><h2>Page finder v3b (lr 3e-4, GPU 1)</h2><div class="big" id="pbp">–</div><div class="bar"><i id="pbb" style="width:0"></i></div><div class="mute" id="pbm"></div></div>
 <div class="card"><h2 id="sdh">SD-HR</h2><div class="big" id="sdp">–</div><div class="bar"><i id="sdb" style="width:0"></i></div><div class="mute" id="sdm"></div></div>
 <div class="card"><h2>Time and cost</h2><div class="big" id="cost">–</div><div class="mute" id="costm"></div></div>
 <div class="card"><h2>Machine</h2><div id="gpu" class="mute"></div></div>
 <div class="card"><h2>Downloads</h2><div id="dlp" class="mute"></div><div style="margin-top:8px">UVDoc photos: <b id="uvp">–</b></div>
  <div class="bar"><i id="uvb" style="width:0"></i></div><div class="mute" id="uvm"></div></div>
</div>
<div class="two">
 <div class="card"><h2>Page finder: training loss</h2><canvas id="c1"></canvas></div>
 <div class="card"><h2>Page finder: validation mIoU (higher is better)</h2><canvas id="c2"></canvas></div>
 <div class="card"><h2>SD-HR: training loss by source</h2><canvas id="c3"></canvas></div>
 <div class="card"><h2>SD-HR: held-out error ×1000 (lower is better)</h2><canvas id="c4"></canvas></div>
</div>
<div class="two">
 <div class="card"><h2>Page finder: latest evaluation</h2><div id="pfe" class="mute">none yet (every 2,500 steps)</div></div>
 <div class="card"><h2>SD-HR: latest evaluation</h2><div id="sde" class="mute">none yet (every 1,000 steps)</div></div>
</div>
<div class="two"><div class="card"><h2>box.log</h2><pre id="box"></pre></div><div class="card"><h2>Trainer output</h2><pre id="tl"></pre></div></div>
</main><script>
const C=['#2f6fe4','#e4572e','#17a673','#9b51e0','#f2a541','#2bb3c0'],charts={};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function hms(s){if(s==null||!isFinite(s))return'–';s=Math.max(0,s);const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?`${h} h ${m} min`:`${m} min`}
function chart(id,datasets,ylog){const d={datasets:datasets.map((s,i)=>({label:s.label,data:s.data,borderColor:C[i%C.length],backgroundColor:C[i%C.length],pointRadius:s.data.length<40?2:0,borderWidth:1.5,tension:.2}))};
 if(charts[id]){charts[id].data=d;charts[id].update('none');return}
 charts[id]=new Chart(document.getElementById(id),{type:'line',data:d,options:{animation:false,parsing:false,maintainAspectRatio:false,
 scales:{x:{type:'linear',title:{display:true,text:'step',color:css('--mute')},ticks:{color:css('--mute')},grid:{color:css('--line')}},
 y:{type:ylog?'logarithmic':'linear',ticks:{color:css('--mute')},grid:{color:css('--line')}}},plugins:{legend:{labels:{color:css('--ink'),boxWidth:10}}}}})}
const pts=(rs,k)=>rs.filter(r=>r[k]!=null).map(r=>({x:r.step,y:r[k]}));
function smooth(p,w=5){return p.map((q,i)=>{const s=p.slice(Math.max(0,i-w+1),i+1);return{x:q.x,y:s.reduce((a,b)=>a+b.y,0)/s.length}})}
function prog(j,p,b,m,what){const f=j.steps?j.step/j.steps:0;document.getElementById(p).textContent=`${j.step.toLocaleString()} / ${j.steps.toLocaleString()}`;
 document.getElementById(b).style.width=(100*f).toFixed(1)+'%';
 document.getElementById(m).textContent=j.stopped?`stopped at step ${j.step.toLocaleString()} (checkpoints kept)`:j.step?`${(100*f).toFixed(1)}% · ${j.rate?j.rate.toFixed(2)+' steps/s':''} · ${hms(j.eta_s)} left · ${j.pushed}/${j.ckpts} checkpoints pushed`:what}
async function tick(){let d;try{d=await (await fetch('api',{cache:'no-store'})).json()}catch(e){document.getElementById('sub').textContent='no answer from the box ('+e+')';return}
 const el=d.started?d.now-d.started:null;document.getElementById('sub').textContent=`updated ${new Date(d.now*1000).toLocaleTimeString()} · refreshes every 5 s`;
 prog(d.pf,'pfp','pfb','pfm','waiting to start');document.getElementById('sdh').textContent=d.sd.steps===20000?'SD-HR pass 5 (GPU 0)':'SD-HR pass 4';prog(d.pfb,'pbp','pbb','pbm','not started');prog(d.sd,'sdp','sdb','sdm','waiting to start (after the UVDoc photos)');
 const left=Math.max(d.pf.eta_s||0,d.pfb.eta_s||0,d.sd.eta_s||0);document.getElementById('cost').textContent=el?`$${(el/3600*d.price).toFixed(2)}`:'–';
 document.getElementById('costm').textContent=`running ${hms(el)} at $${d.price}/h · about ${hms(left)} to go (≈ $${(left/3600*d.price).toFixed(2)} more)`;
 const u=d.dl.uvdoc;document.getElementById('dlp').innerHTML=d.dl.parts.map(([k,v])=>`${v==='done'?'✓':v==='failed'?'✗':'…'} ${k}${v==='done'?'':' <span class="mute">('+v+')</span>'}`).join(' · ');
 document.getElementById('uvp').textContent=`${u.files.toLocaleString()} / ${u.total_files.toLocaleString()} (${u.gb.toFixed(2)} / ${u.total_gb} GB)`;
 document.getElementById('uvb').style.width=(100*u.gb/u.total_gb).toFixed(1)+'%';
 document.getElementById('uvm').textContent=(u.running?(u.rate!=null?`${(u.rate/1e6).toFixed(2)} MB/s`+(u.rate>1e4?` · about ${hms((u.total_gb*1e9-u.gb*1e9)/u.rate)} left`:''):'measuring speed…'):'not downloading')+' · SD-HR trains on the photos already here';
 document.getElementById('gpu').innerHTML=d.gpus.map(g=>g.error?g.error:`GPU ${g.i}: <b>${g.util}%</b> · ${g.mem.toFixed(1)}/${g.mem_total.toFixed(0)} GB · ${g.temp}°C${g.power?` · ${g.power.toFixed(0)} W`:''}`).join('<br>')+
  (d.ram.total?`<br>RAM ${d.ram.used.toFixed(0)}/${d.ram.total.toFixed(0)} GB · `:'<br>')+`disk ${d.disk.used.toFixed(0)}/${d.disk.total.toFixed(0)} GB`;
 chart('c1',[{label:'v3, lr 6e-4 (smoothed)',data:smooth(pts(d.pf.train,'loss'))},{label:'v3b, lr 3e-4 (smoothed)',data:smooth(pts(d.pfb.train,'loss'))}],true);
 const vs=[...new Set(d.pf.evals.flatMap(r=>Object.keys(r.eval||{})))];
 chart('c2',[d.pf,d.pfb].flatMap((j,b)=>vs.map(v=>({label:v.replace('.npz','').replace('val_','')+(b?' (b)':''),data:j.evals.filter(r=>r.eval&&r.eval[v]).map(r=>({x:r.step,y:r.eval[v].miou}))}))));
 chart('c3',['synth_l1','uvdoc_l1','anchor_l1','synth_curve_fg'].map(k=>({label:k,data:smooth(pts(d.sd.train,k))})),true);
 const ks=Object.keys(d.sd.evals.at(-1)||{}).filter(k=>/^syn_|^hr_/.test(k));
 chart('c4',ks.map(k=>({label:k.replace(/^syn_/,''),data:pts(d.sd.evals,k)})));
 const pe=d.pf.evals.at(-1),p0=d.pf.evals[0];if(pe)document.getElementById('pfe').innerHTML=`<table><tr><th>set (step ${pe.step})</th><th>mIoU</th><th>at start</th><th>failures</th></tr>`+
  Object.entries(pe.eval).map(([k,v])=>`<tr><td>${k.replace('.npz','')}</td><td>${v.miou.toFixed(4)}</td><td>${p0&&p0.eval[k]?p0.eval[k].miou.toFixed(4):'–'}</td><td>${v.fail}/${v.n}</td></tr>`).join('')+'</table>';
 const se=d.sd.evals.at(-1),s0=d.sd.evals[0];if(se)document.getElementById('sde').innerHTML=`<table><tr><th>set (step ${se.step})</th><th>now</th><th>at start</th></tr>`+
  ks.map(k=>`<tr><td>${k}</td><td>${se[k].toFixed(2)}</td><td>${s0&&s0[k]!=null?s0[k].toFixed(2):'–'}</td></tr>`).join('')+'</table>';
 document.getElementById('box').textContent=d.box.join('');document.getElementById('tl').textContent='— page finder —\n'+d.pf.log.join('')+'\n— SD-HR —\n'+d.sd.log.join('')}
tick();setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api'):
            body, kind = json.dumps(api()).encode(), 'application/json'
        else:
            body, kind = PAGE.encode(), 'text/html; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.environ.get('WORK', '/root/fs/work'))
    ap.add_argument('--port', type=int, default=17070)
    ap.add_argument('--price', type=float, default=float(os.environ.get('PRICE', 1.2)))
    A = ap.parse_args()
    ThreadingHTTPServer(('127.0.0.1', A.port), H).serve_forever()


if __name__ == '__main__':
    main()
