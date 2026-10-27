"""Fetch only some members of a large remote zip with HTTP range requests (no full download).

  python fetch_zip.py URL OUT_DIR PREFIX[,PREFIX...] [--limit N] [--jobs 16] [--names FILE]

UVDoc_final.zip is 29.5 GB, but the flattening fine-tune needs only img/, grid2d/, metadata_sample/ and
split.json (about a third). --limit N keeps the first N members per prefix (a smoke test); --names FILE fetches exactly those members
(e.g. the grid2d/<geometry>.mat files that the fetched samples' metadata name).
Members already on disk with the right size are skipped, so an interrupted fetch resumes."""
import argparse, os, struct, sys, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor


def get(url, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if k == 4:
                raise


def size_of(url):
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
        return int(r.headers['Content-Length'])


def central_directory(url):
    n = size_of(url)
    tail = get(url, max(0, n - 65558), n - 1)
    i = tail.rfind(b'PK\x05\x06')
    if i < 0:
        raise SystemExit('no end of central directory')
    cd_size, cd_off = struct.unpack('<II', tail[i + 12:i + 20])
    j = tail.rfind(b'PK\x06\x07')   # zip64 locator
    if j >= 0:
        (z64_off,) = struct.unpack('<Q', tail[j + 8:j + 16])
        rec = get(url, z64_off, z64_off + 55)
        cd_size, cd_off = struct.unpack('<QQ', rec[40:56])
    cd = get(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[p + 10:p + 12])
        csize, usize = struct.unpack('<II', cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack('<HHH', cd[p + 28:p + 34])
        off, = struct.unpack('<I', cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):   # zip64 extended information
            hid, hl = struct.unpack('<HH', extra[q:q + 4])
            if hid == 1:
                vals, r = [], q + 4
                for want in (usize == 0xFFFFFFFF, csize == 0xFFFFFFFF, off == 0xFFFFFFFF):
                    if want:
                        vals.append(struct.unpack('<Q', extra[r:r + 8])[0]); r += 8
                it = iter(vals)
                if usize == 0xFFFFFFFF: usize = next(it)
                if csize == 0xFFFFFFFF: csize = next(it)
                if off == 0xFFFFFFFF: off = next(it)
            q += 4 + hl
        out.append((name, method, csize, usize, off))
        p += 46 + nlen + elen + clen
    return out


def fetch(url, m, out_dir, strip):
    name, method, csize, usize, off = m
    rel = name[len(strip):] if strip and name.startswith(strip) else name
    dst = os.path.join(out_dir, rel)
    if os.path.exists(dst) and os.path.getsize(dst) == usize:
        return 0
    hdr = get(url, off, off + 29)
    nlen, elen = struct.unpack('<HH', hdr[26:30])
    data = get(url, off + 30 + nlen + elen, off + 30 + nlen + elen + csize - 1)
    if method == 8:
        data = zlib.decompress(data, -15)
    elif method != 0:
        raise SystemExit(f'{name}: compression method {method}')
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    with open(dst + '.part', 'wb') as f:
        f.write(data)
    os.replace(dst + '.part', dst)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url'); ap.add_argument('out'); ap.add_argument('prefixes')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--names', help='a file of member names (relative to the top folder of the zip) to fetch as well')
    a = ap.parse_args()
    members = [m for m in central_directory(a.url) if not m[0].endswith('/') and not m[0].startswith('__MACOSX')]
    top = os.path.commonprefix([m[0] for m in members])
    strip = top[:top.rfind('/') + 1] if '/' in top else ''
    want = []
    for pre in a.prefixes.split(','):
        sel = [m for m in members if m[0][len(strip):].startswith(pre)]
        sel.sort()
        if a.limit:
            sel = sel[:a.limit]
        print(f'{pre}: {len(sel)} members, {sum(m[2] for m in sel) / 2 ** 30:.2f} GB', flush=True)
        want += sel
    if a.names:
        names = set(open(a.names).read().split())
        sel = [m for m in members if m[0][len(strip):] in names]
        print(f'--names: {len(sel)} of {len(names)} found', flush=True)
        want += sel
    done = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, n in enumerate(ex.map(lambda m: fetch(a.url, m, a.out, strip), want)):
            done += n
            if k % 500 == 0:
                print(f'{k}/{len(want)} {done / 2 ** 30:.2f} GB', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    sys.exit(main())
