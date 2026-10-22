"""Cheap single-core stand-in for the DocUNet LD (for a first look while the heavy lock is busy; the real scorer
decides): both images in gray at the scorer's 598400 px, DIS optical flow (medium) from scan to output, mean
flow length.  python proxy_ld.py VALDIR TAG [TAG ...]"""
import json
import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(1)
v = sys.argv[1]
cr = json.load(open(os.path.join(v, 'crops.json')))
cache = {}
for tag in sys.argv[2:]:
    ld = {}
    for k, c in cr.items():
        if k not in cache:
            g = cv2.imread(c['gt'], cv2.IMREAD_GRAYSCALE)
            s = np.sqrt(598400 / g.size)
            cache[k] = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
        r = cache[k]
        x = cv2.imread(os.path.join(v, tag, f'{k}.png'), cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_AREA)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        f = dis.calc(r, x, None)
        ld[k] = float(np.sqrt((f ** 2).sum(-1)).mean())
    json.dump(ld, open(os.path.join(v, tag, 'proxy_ld.json'), 'w'))
    print(tag, 'proxy LD %.3f' % np.mean(list(ld.values())), flush=True)
