"""RuleNet: a small U-Net that marks the printed straight rules of a stage-1 page (the page the nets
flattened, at half its size: 800 px wide) - logits of [horizontal rule, vertical rule] per pixel,
same size as the input (BGR in [0, 1], sides multiples of 32).

Rules are 1 - 3 px wide at this scale, so a light full-resolution stem keeps them; whether a long
dark run is a rule or a row of text, a curve, a barcode or a picture's edge is decided with context:
the encoder goes to 1/16 with dilated convs and the global mean at the bottom, and a line block at
1/4 and 1/8 looks 15 cells along each axis (depthwise 1x15 and 15x1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class LineBlock(nn.Module):
    """long thin receptive fields along x and y (depthwise), mixed back with a 1x1"""

    def __init__(self, c, k=15):
        super().__init__()
        self.h = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
        self.v = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False)
        self.mix = nn.Sequential(nn.Conv2d(3 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

    def forward(self, x):
        return x + self.mix(torch.cat([x, self.h(x), self.v(x)], 1))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.c = nn.Sequential(cbr(i + skip, o), cbr(o, o))

    def forward(self, x, s):
        x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
        return self.c(torch.cat([x, s], 1))


class RuleNet(nn.Module):
    def __init__(self, ch=(8, 16, 24, 32, 48, 64)):
        super().__init__()
        c0, c1, c2, c3, c4, c5 = ch
        self.stem = cbr(3, c0)                                        # 1/1
        self.e1 = nn.Sequential(cbr(c0, c1, 2), cbr(c1, c1))         # 1/2
        self.e2 = nn.Sequential(cbr(c1, c2, 2), cbr(c2, c2), LineBlock(c2))   # 1/4
        self.e3 = nn.Sequential(cbr(c2, c3, 2), cbr(c3, c3), LineBlock(c3))   # 1/8
        self.e4 = nn.Sequential(cbr(c3, c4, 2), cbr(c4, c4))         # 1/16
        self.ctx = nn.Sequential(cbr(c4, c5, d=2), cbr(c5, c5, d=4))
        self.glob = nn.Sequential(nn.Conv2d(c5, c5, 1), nn.ReLU(inplace=True))
        self.fuse = cbr(2 * c5, c4)
        self.d3 = Up(c4, c3, c3)
        self.d2 = Up(c3, c2, c2)
        self.d1 = Up(c2, c1, c1)
        self.d0 = cbr(c1 + c0 + 3, c0)                               # one light conv at full size
        self.out = nn.Conv2d(c0, 2, 3, 1, 1)
        nn.init.constant_(self.out.bias, -3.0)

    def forward(self, x):
        s = self.stem(x)
        e1 = self.e1(s); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        c = self.ctx(e4)
        g = self.glob(c.mean((2, 3), keepdim=True)).expand_as(c)
        b = self.fuse(torch.cat([c, g], 1))
        d = self.d1(self.d2(self.d3(b, e3), e2), e1)
        d = F.interpolate(d, size=x.shape[2:], mode='bilinear', align_corners=False)
        d = self.d0(torch.cat([d, s, x], 1))
        return self.out(d)   # logits


if __name__ == '__main__':
    import time
    m = RuleNet().eval()
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M params')
    x = torch.rand(1, 3, 1152, 800)
    with torch.no_grad():
        m(x); t = time.time(); y = m(x); print(y.shape, 'cpu %.2f s at 800x1152' % (time.time() - t))
