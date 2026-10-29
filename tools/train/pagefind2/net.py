"""Page finder v2 (round 19): the round-4 SegNet (MobileNetV3-Large encoder + light U-Net decoder) on a letterboxed
input (the photo's long side scaled to SIZE, centred, padded with the ImageNet mean: no aspect squash), 3 outputs:
  0 paper (the page's mask)   1 gutter (book spread's middle line)   2 paper edge (a thin band on the page's outline
  where it is a real edge: not along the photo's own border)
Trained from ImageNet weights only (timm mobilenetv3_large_100, Apache-2.0)."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 384
PAD_BGR = (104, 116, 124)   # the ImageNet mean: zero after normalisation
OUT = 3


def dw(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                         nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegNet2(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        import timm
        self.enc = timm.create_model('mobilenetv3_large_100', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3, 4))
        if pretrained:
            self.enc.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)
        ch = self.enc.feature_info.channels()
        D = 64
        self.lat = nn.ModuleList([nn.Sequential(nn.Conv2d(c, D, 1, bias=False), nn.BatchNorm2d(D), nn.ReLU(inplace=True)) for c in ch])
        self.ctx = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch[-1], D, 1), nn.ReLU(inplace=True))
        self.fuse = nn.ModuleList([dw(2 * D, D) for _ in range(4)])
        self.head = nn.Sequential(dw(D + 3, 32), nn.Conv2d(32, OUT, 1))

    def forward(self, x):
        f = self.enc(x)
        y = self.lat[4](f[4]) + self.ctx(f[4])
        for i in (3, 2, 1, 0):
            y = F.interpolate(y, size=f[i].shape[-2:], mode='bilinear', align_corners=False)
            y = self.fuse[i](torch.cat([y, self.lat[i](f[i])], 1))
        y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(torch.cat([y, x], 1))


def letterbox(img, size=SIZE, interp=cv2.INTER_AREA, pad=PAD_BGR):
    """-> (S x S image, (scale, x0, y0)): photo px p -> letterbox px p * scale + (x0, y0)"""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.empty((size, size) + img.shape[2:], img.dtype)
    out[...] = pad if img.ndim == 3 else 0
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out, (s, x0, y0)


def to_input(bgr):
    x, tr = letterbox(bgr)
    x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), tr
