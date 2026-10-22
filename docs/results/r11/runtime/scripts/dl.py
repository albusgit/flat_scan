import sys, zipfile, glob, os, collections, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import cat
for spec in sys.argv[1:]:
    tag, pat = spec.split("=", 1)
    c = collections.Counter()
    for f in glob.glob(pat):
        z = zipfile.ZipFile(f)
        for i in z.infolist():
            c[cat(i.filename)] += len(zlib.compress(z.read(i), 9))
    print(f"== {tag} deflated sum {sum(c.values())/1e6:.2f} MB")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        if v > 2e5: print(f"  {k:45s} {v/1e6:7.2f}")
