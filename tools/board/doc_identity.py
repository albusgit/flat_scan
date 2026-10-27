"""The document path is untouched by whiteboard mode: every photo goes through two pipeline_cli builds (the
commit before and the one after, app settings: nets, rule net, shadow net) with the finger erase off (its
fill changed on purpose, feature 5), and the flattened pages (lossless PNG), the rendered pages and the
JSON (timings left out) are compared byte for byte. --fingers: with the finger erase on (as the app runs;
photos differ where either build fills a finger).
Usage: python3 -I tools/board/doc_identity.py [--fingers] BASE_CLI NEW_CLI OUTDIR PHOTO... -> one line per photo, a summary"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

args = [x for x in sys.argv[1:] if x != "--fingers"]
fingers = len(args) != len(sys.argv) - 1
base, new, out = args[:3]
photos = args[3:]
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
models = os.path.join(root, "app", "src", "main", "assets", "models")
env = dict(os.environ, FS_MODELS=models, FS_NETS="1", FS_RULE_NET="1", FS_SHADE_NET=os.path.join(models, "shadenet.onnx"), FS_RENDER="1")
if not fingers:
    env["FS_NO_FINGERS"] = "1"
os.makedirs(out, exist_ok=True)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def run(cli, tag, photo):
    b = os.path.join(out, tag + "_" + os.path.splitext(os.path.basename(photo))[0])
    r = subprocess.run([cli, photo, b + ".jpg", "net"], env=dict(env, FS_PAGE_PNG=b), capture_output=True, text=True, timeout=900)
    try:
        j = json.loads(r.stdout.strip().splitlines()[-1])
        for k in ("nets", "pipeline_ms", "ms", "photo"):
            j.pop(k, None)
        js = json.dumps(j, sort_keys=True)
    except Exception:
        js = None
    return md5(b + "_page.png"), md5(b + ".jpg.auto.jpg"), js


def one(photo):
    a = run(base, "base", photo)
    b = run(new, "new", photo)
    return os.path.basename(photo), a == b and a[0] is not None, a[0] is not None


res = list(ThreadPoolExecutor(2).map(one, photos))
for n, same, ok in res:
    print(f"{'IDENTICAL' if same else ('DIFFERS' if ok else 'FAILED')} {n}")
print(f"{sum(1 for _, s, _ in res if s)} / {len(res)} identical (page PNG, rendered page, JSON)")
