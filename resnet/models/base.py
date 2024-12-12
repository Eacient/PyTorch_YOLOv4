from models.common import *
import torch.nn.functional as F

class ReLUConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ReLUConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, s=1):
        super(ResBlock, self).__init__()
        self.conv1 = ReLUConv(c1, c2, 3, s)
        self.conv2 = ReLUConv(c2, c2, 3, 1, g=g, act=False)
        if shortcut:
            if c1 != c2 or s != 1:
                self.conv_path = ReLUConv(c1, c2, k=1, s=s, act=False)
            else:
                self.conv_path = nn.Identity()
        self.add = shortcut
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.conv_path(x)+self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x)))

class ResBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, s=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ReLUConv(c1, c_, 1, 1)
        self.conv2 = ReLUConv(c_, c_, 3, s, g=g)
        self.conv3 = ReLUConv(c_, c2, 1, 1, act=False)
        if shortcut:
            if c1 != c2 and s == 1:
                self.conv_path = ReLUConv(c1, c2, k=1, s=s, act=False)
            else:
                self.conv_path = nn.Identity()
        self.add = shortcut
        self.act = nn.ReLU

    def forward(self, x):
        return self.act(self.conv_path(x) + self.conv3(self.conv2(self.conv1(x))) if self.add else self.conv3(self.conv2(self.conv1(x))))

class ResLayer(nn.Module):
    def __init__(self, c1, c2, n=1, s=1, shortcut=True, g=1):
        super(ResLayer, self).__init__()
        blocks = [ResBlock(c1, c2, shortcut, g, s)]
        blocks += [ResBlock(c2, c2, shortcut, g) for _ in range(n-1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class ResBottleneckLayer(nn.Module):
    def __init__(self, c1, c2, n=1, s=1, shortcut=True, g=1, e=0.25):
        super(ResBottleneckLayer, self).__init__()
        blocks = [ResBottleneck(c1, c2, shortcut, g, e, s)]
        blocks += [ResBottleneck(c2, c2, shortcut, g, e) for _ in range(n-1)]
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)