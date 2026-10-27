"""Shared helpers for the flattening-net fine-tuning (Round 11, tools/flatnet).

SyntheticDoc high-res (SD-HR, Woortmann/Magne/Sorkine-Hornung, ECCV 2026; code MIT, sd_model.py is their
training/model.py unchanged): input 1x3x1440x1024 RGB in [-1, 1] (the app's ONNX wrapper takes [0, 1]),
output bm_map 1x2x121x177: grid_sample coordinates (x, y) in [-1, 1], align_corners=True, of the input
image for each point of the flat page (rows = page top->bottom, 121 samples; cols = page left->right, 177
samples; the page is simply stretched to the grid)."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sd_model import UVDocNet  # noqa: E402

SD_CKPT = os.environ.get('FS_SDHR_CKPT', r'C:\Users\User\Desktop\model_zoo\SyntheticDoc\weights\highres_121x177.ckpt')
IN_H, IN_W = 1440, 1024
GH, GW = 121, 177


def load_sdhr(path=SD_CKPT, device='cuda'):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    sd = ck.get('state_dict', ck)
    sd = {(k.replace('model.', '', 1) if k.startswith('model.') else k): v for k, v in sd.items()}
    net = UVDocNet(32, 5, (GH, GW), (IN_H, IN_W))
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def bm_only_forward(net, x):
    """BM head only (skips the shadow decoder, which the app does not use): same values as net(x)['bm_map']."""
    s1 = net.enc_s1(x)
    s2 = net.enc_s2(s1)
    s3 = net.enc_s3(s2)
    s4 = net.enc_s4(s3)
    feat = net.enc_s5(s4)
    bridge = net.bridge_concat(torch.cat([net.bridge_1(feat), net.bridge_2(feat), net.bridge_3(feat),
                                          net.bridge_4(feat), net.bridge_5(feat), net.bridge_6(feat)], dim=1))
    return F.interpolate(net.bm_head(bridge), size=(GH, GW), mode='bilinear', align_corners=False)


def to_input(img_rgb_u8):
    """HxWx3 uint8 RGB (already IN_H x IN_W) -> 1x3xHxW float in [-1, 1]."""
    t = torch.from_numpy(img_rgb_u8).permute(2, 0, 1).float().div(255.0)
    return (t * 2 - 1)[None]


@torch.no_grad()
def predict_bm(net, img_rgb_u8, device='cuda'):
    """img (any size) -> bm 2xGHxGW in [-1, 1] coords of that image."""
    import cv2
    x = cv2.resize(img_rgb_u8, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    with torch.autocast('cuda', dtype=torch.float16, enabled=False):
        bm = bm_only_forward(net, to_input(x).to(device))
    return bm[0].float().cpu().numpy()


def bm_to_px(bm, w, h):
    """bm in [-1,1] (align_corners=True) -> pixel coords in an image of size w x h. bm: 2xGHxGW."""
    out = np.empty_like(bm)
    out[0] = (bm[0] + 1) * 0.5 * (w - 1)
    out[1] = (bm[1] + 1) * 0.5 * (h - 1)
    return out


def px_to_bm(px, w, h):
    out = np.empty_like(px)
    out[0] = px[0] / (w - 1) * 2 - 1
    out[1] = px[1] / (h - 1) * 2 - 1
    return out


def upsample_field(f, oh, ow):
    """2xGHxGW float field -> 2 x oh x ow, bilinear align_corners=True (as the SD unwarp)."""
    t = torch.from_numpy(np.ascontiguousarray(f))[None].float()
    return F.interpolate(t, size=(oh, ow), mode='bilinear', align_corners=True)[0].numpy()


def unwarp_px(img, map_px, oh, ow):
    """img HxWxC, map_px 2xGHxGW pixel coords of img -> oh x ow unwarped (cv2.remap, bilinear)."""
    import cv2
    m = upsample_field(map_px, oh, ow)
    return cv2.remap(img, m[0].astype(np.float32), m[1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
