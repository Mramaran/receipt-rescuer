"""A small NAFNet-style restoration network for grayscale receipts.

Architecture follows Chen et al., "Simple Baselines for Image Restoration"
(ECCV 2022): activation-free blocks with a channel-splitting gate and
simplified channel attention, in a U-shaped encoder/decoder. Written from
scratch for this project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Layer norm over the channel axis of an NCHW tensor."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        y = F.layer_norm(x.permute(0, 2, 3, 1), self.weight.shape, self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv2 = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, c * 2, 1)
        self.conv4 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.gate(self.dw(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        y = self.conv4(self.gate(self.conv3(self.norm2(x))))
        return x + y * self.gamma


class RescueNet(nn.Module):
    # One block per side at full resolution keeps memory low; depth sits at 1/4 and 1/8 scale.
    def __init__(self, width=32, enc_blocks=(1, 2, 6), mid_blocks=4, dec_blocks=(2, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 1, 3, padding=1)
        self.encs, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decs, self.ups = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.stride = 2 ** len(enc_blocks)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        inp = F.pad(x, (0, pw, 0, ph), mode="replicate")
        y = self.intro(inp)
        skips = []
        for enc, down in zip(self.encs, self.downs):
            y = enc(y)
            skips.append(y)
            y = down(y)
        y = self.mid(y)
        for dec, up, skip in zip(self.decs, self.ups, reversed(skips)):
            y = dec(up(y) + skip)
        out = self.ending(y) + inp
        return out[..., :h, :w]


def count_params(model):
    return sum(p.numel() for p in model.parameters())
