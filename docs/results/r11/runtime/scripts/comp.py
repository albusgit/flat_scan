import sys, zipfile, glob, os, collections
def cat(n):
    if n.startswith("lib/"): return "lib " + os.path.basename(n)
    if n.endswith(".dex"): return "dex"
    if n.endswith((".tflite", ".onnx", ".bin", ".ort")) or "/models/" in n: return "model " + os.path.basename(n)
    if n.startswith("assets/"): return "assets other"
    if n.startswith("res/") or n == "resources.arsc": return "res"
    return "other"
def main():
  pass
for spec in (sys.argv[1:] if __name__ == "__main__" else []):
    tag, pat = spec.split("=", 1)
    c = collections.Counter(); tot = 0
    for f in glob.glob(pat):
        for i in zipfile.ZipFile(f).infolist():
            c[cat(i.filename)] += i.compress_size
        tot += os.path.getsize(f)
    print(f"== {tag} files {tot/1e6:.2f} MB")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        if v > 2e5: print(f"  {k:45s} {v/1e6:7.2f}")
