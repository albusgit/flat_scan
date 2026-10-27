"""run2_core.py: SD-HR's forward with the coordinate head in float32, the run-2 losses, and per-batch source quotas.
(From the round-18 outside review, checked here: tools/train/flat2/README notes.)

bm_forward    the encoder and bridge under bf16 autocast, the BM head and its upsampling in float32. Under autocast the
              head's convolutions ran in bf16: on 200 held synthetic samples the 36k checkpoint's derivative error was
              0.063 in bf16 against 0.036 in float32 (as the app runs it), so most of what the derivative and curvature
              losses saw was rounding.
endpoint_resize  a coordinate field resized with its end points on the end points (align_corners), as the app reads a
              grid (uvdoc.cpp UvGrid::at). cv2.resize puts pixel centres there instead: up to 2.4 px off at 1024.
map_terms     per-sample terms; target points outside [-1, 1] or not finite are masked out (never clamped inward).
mixed_loss    synthetic / UVDoc samples: L1 + 0.25 derivative + 0.10 target curvature (flat kinds) + 1.0 border
              (one mean over the perimeter) + 2.0 cell-area floor (each cell's four corner Jacobians at least 5 % of
              the target's, same sign: a collapsed or folded cell costs). Anchor samples (the original net as
              teacher): L1 + 0.10 derivative + 0.5 border.
BalancedBatches  exactly counts[source] samples of each source per batch, indices (source, global id)."""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

GH, GW = 121, 177
FLAT_KINDS = (0, 1)   # Mixed.KINDS: flat, nearflat


def endpoint_resize(field, size):
    """C x H x W or N x C x H x W coordinate field -> size (h, w), end points kept"""
    single = field.ndim == 3
    out = F.interpolate(field[None] if single else field, size=size, mode='bilinear', align_corners=True)
    return out[0] if single else out


def bm_forward(net, x, fp32_head=True):
    """SD-HR's BM map (2 x 121 x 177, the same values as common.bm_only_forward in float32). fp32_head=False: the
    whole forward under bf16 autocast, as run 1 trained (a control only)."""
    cuda = x.is_cuda
    amp = (lambda on: torch.autocast('cuda', dtype=torch.bfloat16, enabled=on)) if cuda else (lambda on: contextlib.nullcontext())
    with amp(True):
        z = x
        for name in ('enc_s1', 'enc_s2', 'enc_s3', 'enc_s4', 'enc_s5'):
            z = getattr(net, name)(z)
        z = net.bridge_concat(torch.cat([getattr(net, f'bridge_{i}')(z) for i in range(1, 7)], dim=1))
        if not fp32_head:
            return F.interpolate(net.bm_head(z), size=(GH, GW), mode='bilinear', align_corners=False).float()
    with amp(False):
        return F.interpolate(net.bm_head(z.float()), size=(GH, GW), mode='bilinear', align_corners=False)


def masked_mean(x, m):
    m = m.to(x.dtype).expand_as(x)
    return (x * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1)


def quad_dets(g):
    """the Jacobian determinant at each cell's four corners (grid units: the identity map gives about 1)"""
    h, w = g.shape[-2:]
    top = (g[:, :, :-1, 1:] - g[:, :, :-1, :-1]) * ((w - 1) / 2)
    bot = (g[:, :, 1:, 1:] - g[:, :, 1:, :-1]) * ((w - 1) / 2)
    left = (g[:, :, 1:, :-1] - g[:, :, :-1, :-1]) * ((h - 1) / 2)
    right = (g[:, :, 1:, 1:] - g[:, :, :-1, 1:]) * ((h - 1) / 2)
    cr = lambda a, b: a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return torch.stack([cr(top, left), cr(top, right), cr(bot, left), cr(bot, right)], 1)


def map_terms(p, t):
    """p, t: B x 2 x GH x GW -> dict of per-sample terms (float32)"""
    p, t = p.float(), t.float()
    finite = torch.isfinite(t).all(1, keepdim=True)
    valid = finite & (t.abs() <= 1).all(1, keepdim=True)
    t = torch.where(finite.expand_as(t), t, torch.zeros_like(t))
    h, w = p.shape[-2:]
    e = (p - t).abs()
    border = torch.zeros_like(valid)
    border[:, :, 0] = border[:, :, -1] = True
    border[:, :, :, 0] = border[:, :, :, -1] = True
    pu, pv = torch.diff(p, dim=3) * ((w - 1) / 2), torch.diff(p, dim=2) * ((h - 1) / 2)
    tu, tv = torch.diff(t, dim=3) * ((w - 1) / 2), torch.diff(t, dim=2) * ((h - 1) / 2)
    mu, mv = valid[..., 1:] & valid[..., :-1], valid[:, :, 1:] & valid[:, :, :-1]
    deriv = masked_mean((pu - tu).abs(), mu) + masked_mean((pv - tv).abs(), mv)
    curve = (masked_mean(torch.diff(pu - tu, dim=3).abs(), mu[..., 1:] & mu[..., :-1]) +
             masked_mean(torch.diff(pv - tv, dim=2).abs(), mv[:, :, 1:] & mv[:, :, :-1]))
    cp, ct = quad_dets(p), quad_dets(t)
    cm = valid[:, :, :-1, :-1] & valid[:, :, :-1, 1:] & valid[:, :, 1:, :-1] & valid[:, :, 1:, 1:]
    ok = cm & (ct.abs() > 1e-3)
    area = masked_mean(F.relu(0.05 - cp * ct.sign() / ct.abs().clamp_min(1e-3)), ok)
    return dict(l1=masked_mean(e, valid), deriv=deriv, curve=curve, border=masked_mean(e, valid & border), area=area)


def mixed_loss(p, t, kind, anchor):
    """mean over the batch of the per-sample objective; returns (loss, terms)"""
    q = map_terms(p, t)
    flat = sum((kind == k) for k in FLAT_KINDS).float()
    sup = q['l1'] + 0.25 * q['deriv'] + 0.10 * flat * q['curve'] + q['border'] + 2.0 * q['area']
    kd = q['l1'] + 0.10 * q['deriv'] + 0.5 * q['border']
    return torch.where(anchor.bool(), kd, sup).mean(), q


class BalancedBatches(Sampler):
    """batches of exactly counts[source] samples per source, as (source, id) with ids unique over the run (the
    dataset draws sample `id` of `source` from a generator seeded by it: resumable, no epoch state in workers)"""

    def __init__(self, counts, steps, start=0, seed=0):
        self.src = [s for s, n in counts.items() for _ in range(n)]
        if not self.src:
            raise ValueError('empty batch')
        self.steps, self.start, self.seed = steps, start, seed

    def __len__(self):
        return max(0, self.steps - self.start)

    def __iter__(self):
        b = len(self.src)
        for step in range(self.start, self.steps):
            order = np.random.default_rng((self.seed, step)).permutation(b)
            yield [(self.src[int(k)], step * b + j) for j, k in enumerate(order)]
