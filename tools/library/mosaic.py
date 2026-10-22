"""Numbered mosaics of a set for hand curation: python -I mosaic.py SET OUT_PREFIX [--per 40] [--cols 8]
Tile k of mosaic m is item m*per + k of sets/SET.json (number printed on the tile)."""
import argparse
import json

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("set")
ap.add_argument("out")
ap.add_argument("--per", type=int, default=40)
ap.add_argument("--cols", type=int, default=8)
a = ap.parse_args()
items = json.load(open(f"E:/fs_datasets/sets/{a.set}.json"))["items"]
W, H = 240, 180
for m in range(0, len(items), a.per):
    tiles = []
    for k, it in enumerate(items[m:m + a.per]):
        im = cv2.imread(it["photo"], cv2.IMREAD_REDUCED_COLOR_4)
        t = np.full((H, W, 3), 128, np.uint8)
        if im is not None:
            s = min(W / im.shape[1], H / im.shape[0])
            r = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            t[:r.shape[0], :r.shape[1]] = r
        cv2.rectangle(t, (0, 0), (46, 26), (0, 0, 0), -1)
        cv2.putText(t, str(m + k), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % a.cols:
        tiles.append(np.zeros((H, W, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
    cv2.imwrite(f"{a.out}_{m // a.per:02d}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
