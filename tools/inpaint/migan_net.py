"""MI-GAN's inference generator (the bare net: no crop / resize / blend pipeline), for tools/inpaint/export_onnx.py.

A copy of lib/model_zoo/migan_inference.py of https://github.com/Picsart-AI-Research/MI-GAN (commit on main as of
2026-10), kept here so the export runs no downloaded code. Changes, none of them numerical:
  * the noise of SynthesisBlock's convs is added as a [1, 1, R, R] tensor (the same broadcast over channels as the
    original [R, R] one; an explicit NCHW rank for the ONNX -> TFLite layout pass), and not in place;
  * lrelu_agc's leaky_relu is not in place.
The state dict keys and shapes are the original ones: migan_512_places2.pt loads with strict=True.

Original copyright and licence (MIT):

    MIT License

    Copyright (c) 2024 Picsart AI Research (PAIR)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Input x [1, 4, R, R] (NCHW): channel 0 = m - 0.5, channels 1..3 = RGB * m, where m is 1 on known pixels and 0 in
the hole, RGB in [-1, 1] (pixel * 2 / 255 - 1). Output [1, 3, R, R]: RGB, ~[-1, 1] (not clamped)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class lrelu_agc:
    """The lrelu layer with alpha, gain and clamp."""

    def __init__(self, alpha=0.2, gain=1, clamp=None):
        self.alpha = alpha
        self.gain = np.sqrt(2) if gain == 'sqrt_2' else gain
        self.clamp = clamp

    def __call__(self, x, gain=1):
        x = F.leaky_relu(x, negative_slope=self.alpha)
        act_gain = self.gain * gain
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x


def setup_filter(f, normalize=True, gain=1):
    f = torch.as_tensor(f, dtype=torch.float32)
    f = f.ger(f)   # [1, 3, 3, 1] -> its 4 x 4 outer product (not separable: fewer than 8 taps)
    if normalize:
        f /= f.sum()
    return f * (gain ** (f.ndim / 2))


class Downsample2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.filter = nn.Conv2d(in_channels, in_channels, kernel_size=4, groups=in_channels, padding=1, bias=False,
                                stride=2)
        f = setup_filter([1, 3, 3, 1], gain=1)
        self.filter.weight = nn.Parameter(f.repeat([*self.filter.weight.shape[:2], 1, 1]))

    def forward(self, x):
        return self.filter(x)


class Upsample2d(nn.Module):
    def __init__(self, in_channels, resolution=None):
        super().__init__()
        self.nearest_up = nn.Upsample(scale_factor=2, mode='nearest')
        w = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float32)
        assert resolution is not None
        self.register_buffer('filter_const', w.repeat(1, 1, resolution // 2, resolution // 2))
        self.filter = nn.Conv2d(in_channels, in_channels, kernel_size=4, groups=in_channels, bias=False)
        f = setup_filter([1, 3, 3, 1], gain=4)
        self.filter.weight = nn.Parameter(f.repeat([*self.filter.weight.shape[:2], 1, 1]))

    def forward(self, x):
        x = self.nearest_up(x)
        x = x * self.filter_const
        x = F.pad(x, pad=(2, 1, 2, 1))
        return self.filter(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, activation=None, resolution=None,
                 use_noise=False, down=1, up=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias,
                               groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, groups=1)
        self.downsample = Downsample2d(in_channels) if down > 1 else None
        self.upsample = Upsample2d(out_channels, resolution=resolution) if up > 1 else None
        self.use_noise = use_noise
        if use_noise:
            assert resolution is not None
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = self.conv2(x)
        if self.upsample is not None:
            x = self.upsample(x)
        if self.use_noise:
            x = x + (self.noise_const * self.noise_strength)[None, None]
        if self.activation is not None:
            x = self.activation(x)
        return x


ACT = lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=256)


class EncoderBlock(nn.Module):
    def __init__(self, ic_n, oc_n, rgb_n=None, activation=ACT, down=2):
        super().__init__()
        self.fromrgb = nn.Conv2d(rgb_n, ic_n, 1) if rgb_n is not None else None
        self.conv1 = SeparableConv2d(ic_n, ic_n, 3, activation=activation)
        self.conv2 = SeparableConv2d(ic_n, oc_n, 3, activation=activation, down=down)
        self.activation = activation

    def forward(self, x, img):
        if self.fromrgb is not None:
            y = self.activation(self.fromrgb(img))
            x = x + y if x is not None else y
        feat = self.conv1(x)
        x = self.conv2(feat)
        return x, feat


class Encoder(nn.Module):
    def __init__(self, resolution=256, ic_n=4, ch_base=32768, ch_max=512, activation=ACT):
        super().__init__()
        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        self.encode_res = [2 ** i for i in range(log2res, 1, -1)]
        self.ic_n = ic_n
        for idx, (resi, resj) in enumerate(zip(self.encode_res[:-1], self.encode_res[1:])):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)
            block = EncoderBlock(hidden_ch_i, hidden_ch_j, rgb_n=ic_n if idx == 0 else None, activation=activation)
            setattr(self, 'b{}'.format(resi), block)
        hidden_ch = min(ch_base // self.encode_res[-1], ch_max)
        self.b4 = EncoderBlock(hidden_ch, hidden_ch, activation=activation, down=1)

    def forward(self, img):
        x = None
        feats = {}
        for resi in self.encode_res[0:-1]:
            x, feat = getattr(self, 'b{}'.format(resi))(x, img)
            feats[resi] = feat
        x, feat = self.b4(x, img)
        feats[4] = feat
        return x, feats


class SynthesisBlockFirst(nn.Module):
    def __init__(self, oc_n, resolution, rgb_n=None, activation=ACT):
        super().__init__()
        self.resolution = resolution
        self.conv1 = SeparableConv2d(oc_n, oc_n, 3, activation=activation)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=4, activation=activation)
        if rgb_n is not None:
            self.torgb = nn.Conv2d(oc_n, rgb_n, 1)

    def forward(self, x, enc_feat):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)
        img = self.torgb(x) if self.torgb is not None else None
        return x, img


class SynthesisBlock(nn.Module):
    def __init__(self, ic_n, oc_n, resolution, rgb_n, activation=ACT):
        super().__init__()
        self.resolution = resolution
        self.conv1 = SeparableConv2d(ic_n, oc_n, 3, resolution=resolution, up=2, activation=activation, use_noise=True)
        self.conv2 = SeparableConv2d(oc_n, oc_n, 3, resolution=resolution, up=1, activation=activation, use_noise=True)
        self.torgb = nn.Conv2d(oc_n, rgb_n, 1) if rgb_n is not None else None
        self.upsample = Upsample2d(rgb_n, resolution=resolution)

    def forward(self, x, enc_feat, img):
        x = self.conv1(x)
        x = x + enc_feat
        x = self.conv2(x)
        if img is not None:
            img = self.upsample(img)
        if self.torgb is not None:
            y = self.torgb(x)
            img = img + y if img is not None else y
        return x, img


class Synthesis(nn.Module):
    def __init__(self, resolution=256, rgb_n=3, ch_base=32768, ch_max=512, activation=ACT):
        super().__init__()
        log2res = int(np.log2(resolution))
        if 2 ** log2res != resolution:
            raise ValueError
        block_res = [2 ** i for i in range(2, log2res + 1)]
        self.resolution = resolution
        self.rgb_n = rgb_n
        self.block_res = block_res
        hidden_ch = min(ch_base // block_res[0], ch_max)
        self.b4 = SynthesisBlockFirst(hidden_ch, resolution=4, rgb_n=rgb_n, activation=activation)
        for resi, resj in zip(block_res[:-1], block_res[1:]):
            hidden_ch_i = min(ch_base // resi, ch_max)
            hidden_ch_j = min(ch_base // resj, ch_max)
            setattr(self, 'b{}'.format(resj),
                    SynthesisBlock(hidden_ch_i, hidden_ch_j, resolution=resj, rgb_n=rgb_n, activation=activation))

    def forward(self, x, enc_feats):
        x, img = self.b4(x, enc_feats[4])
        for res in self.block_res[1:]:
            x, img = getattr(self, f'b{res}')(x, enc_feats[res], img)
        return img


class Generator(nn.Module):
    def __init__(self, resolution=256):
        super().__init__()
        self.synthesis = Synthesis(resolution=resolution)
        self.encoder = Encoder(resolution=resolution)

    def forward(self, x):
        """x: [1, 4, R, R], (m - 0.5, RGB * m); returns RGB [1, 3, R, R]."""
        x, feats = self.encoder(x)
        return self.synthesis(x, feats)
