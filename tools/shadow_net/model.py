"""ShadeNet: a small U-Net that predicts a flat page's shading map (BGR in (0, 1], same size as
its input) from the page (BGR in [0, 1], sides multiples of 32). The page divided by the map is the
page without its shadows. Encoder at 1/2 .. 1/32 with global context at the bottom (dilated convs
plus the global mean, so a whole-page decision - "this grey area is a shadow, that one is a box" -
can see the whole page), decoder back to 1/2, and a light full-resolution head on the input so
shadow edges stay sharp."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.c = nn.Sequential(cbr(i + skip, o), cbr(o, o))

    def forward(self, x, s):
        x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
        return self.c(torch.cat([x, s], 1))


class ShadeNet(nn.Module):
    def __init__(self, ch=(16, 24, 32, 48, 64, 96)):
        super().__init__()
        c1, c2, c3, c4, c5, c6 = ch
        self.e1 = nn.Sequential(cbr(3, c1, 2), cbr(c1, c1))          # 1/2
        self.e2 = nn.Sequential(cbr(c1, c2, 2), cbr(c2, c2))         # 1/4
        self.e3 = nn.Sequential(cbr(c2, c3, 2), cbr(c3, c3))         # 1/8
        self.e4 = nn.Sequential(cbr(c3, c4, 2), cbr(c4, c4))         # 1/16
        self.e5 = nn.Sequential(cbr(c4, c5, 2), cbr(c5, c5))         # 1/32
        self.ctx = nn.Sequential(cbr(c5, c6, d=2), cbr(c6, c6, d=4))
        self.glob = nn.Sequential(nn.Conv2d(c6, c6, 1), nn.ReLU(inplace=True))
        self.fuse = cbr(2 * c6, c5)
        self.d4 = Up(c5, c4, c4)
        self.d3 = Up(c4, c3, c3)
        self.d2 = Up(c3, c2, c2)
        self.d1 = Up(c2, c1, c1)
        self.out1 = nn.Conv2d(c1, 3, 3, 1, 1)
        self.head = nn.Sequential(nn.Conv2d(3 + 3 + c1, 12, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(12, 3, 3, 1, 1))
        nn.init.constant_(self.out1.bias, 3.0)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3); e5 = self.e5(e4)
        c = self.ctx(e5)
        g = self.glob(c.mean((2, 3), keepdim=True)).expand_as(c)
        b = self.fuse(torch.cat([c, g], 1))
        d = self.d1(self.d2(self.d3(self.d4(b, e4), e3), e2), e1)
        z = self.out1(d)                                                     # logits at 1/2
        zu = F.interpolate(z, size=x.shape[2:], mode='bilinear', align_corners=False)
        du = F.interpolate(d, size=x.shape[2:], mode='bilinear', align_corners=False)
        z = zu + self.head(torch.cat([x, zu, du], 1))
        return torch.sigmoid(z)


if __name__ == '__main__':
    m = ShadeNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M params')
    print(m(torch.rand(1, 3, 512, 384)).shape)
