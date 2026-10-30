"""orientnet: which way up is a flattened page. Shared by train.py, eval.py and the C++ port (core/src/orient.cpp
orient_net_*): keep the patch selection here and there the same.

A page is turned to grey, scaled so its long side is LONG px, and cut into PATCH x PATCH patches on a grid
(stride STRIDE); patches whose ink share is within [INK_MIN, INK_MAX] are text-like, the INK-richest MAX_PATCHES
of them go to the net. The net gives 4 logits per patch: the page turned by k x 90 degrees clockwise is upright
(k = 0: as it is; 1: turn it 90 cw; 2: 180; 3: 90 ccw). The page's answer: the mean log-softmax over its patches.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

PATCH = 128
LONG = (1024, 1536)      # two scales: small and large print
STRIDE = 96
INK_MIN, INK_MAX = 0.03, 0.45
MAX_PATCHES = 64


def ink_mask(g):
    """dark marks on the local background (grey uint8)"""
    bg = cv2.medianBlur(cv2.dilate(g, np.ones((5, 5), np.uint8)), 21)
    return (bg.astype(np.int16) - g.astype(np.int16)) > 30


def page_patches(gray):
    """the text-like patches of a page, uint8 N x PATCH x PATCH (both scales)"""
    out, score = [], []
    for L in LONG:
        s = L / max(gray.shape)
        g = cv2.resize(gray, (max(PATCH, int(gray.shape[1] * s)), max(PATCH, int(gray.shape[0] * s))),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        ink = ink_mask(g)
        H, W = g.shape
        for y in range(0, max(1, H - PATCH + 1), STRIDE):
            for x in range(0, max(1, W - PATCH + 1), STRIDE):
                f = ink[y:y + PATCH, x:x + PATCH].mean()
                if INK_MIN <= f <= INK_MAX:
                    out.append(g[y:y + PATCH, x:x + PATCH])
                    score.append(f)
    if not out:
        return np.zeros((0, PATCH, PATCH), np.uint8)
    order = np.argsort(score)[::-1][:MAX_PATCHES]
    return np.stack([out[i] for i in order])


def to_input(p):
    """uint8 N x P x P -> float N x 1 x P x P, each patch normalised to its own mean / spread (lighting)"""
    x = p.astype(np.float32) / 255.0
    m = x.mean(axis=(1, 2), keepdims=True)
    sd = x.std(axis=(1, 2), keepdims=True) + 0.05
    return ((x - m) / sd)[:, None]


class OrientNet(nn.Module):
    def __init__(self, w=(16, 32, 64, 96, 128)):
        super().__init__()
        layers, c = [], 1
        for i, k in enumerate(w):
            layers += [nn.Conv2d(c, k, 3, padding=1, bias=False), nn.BatchNorm2d(k), nn.ReLU(inplace=True),
                       nn.Conv2d(k, k, 3, padding=1, bias=False), nn.BatchNorm2d(k), nn.ReLU(inplace=True)]
            if i < len(w) - 1:
                layers.append(nn.MaxPool2d(2))
            c = k
        self.f = nn.Sequential(*layers)
        self.head = nn.Linear(c, 4)

    def forward(self, x):
        return self.head(self.f(x).mean(dim=(2, 3)))


def widths_of(state_dict):
    """the channels per stage of a saved OrientNet (two convolutions per stage)"""
    return tuple(v.shape[0] for k, v in state_dict.items() if k.startswith('f.') and v.dim() == 4)[::2]


def decide(logp_mean, min_margin):
    """the page's turn k (0..3) and its margin over the runner-up; k = 0 when the margin is below min_margin"""
    o = np.argsort(logp_mean)[::-1]
    margin = float(logp_mean[o[0]] - logp_mean[o[1]])
    return (int(o[0]) if margin >= min_margin else 0), margin
