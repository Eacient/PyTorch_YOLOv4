from models.common import *

class MSE(nn.Module):
    def forward(self, x):
        return nn.functional.mse_loss(x[0], x[1], reduction='none')

class Cosine(nn.Module):
    def forward(self, x):
        return nn.functional.cosine_embedding_loss(x[0].permute(0,2,3,1).flatten(0,2), x[1].permute(0,2,3,1).flatten(0,2), torch.ones(x[0].shape[0]*x[0].shape[2]*x[0].shape[3], device=x[0].device), reduction='none')

class Identity(nn.Module):
    def forward(self, x):
        return x

class ABNChunk(nn.Module):
    def forward(self, x):
        if self.training:
            return torch.chunk(x, 2, 0)[0]
        else:
            return x
    
class FrozeChunk(nn.Module):
    def forward(self, x):
        return torch.chunk(x, 2, 0)[1].detach()

class Repeat(nn.Module):
    def forward(self, x):
        return torch.repeat_interleave(x, 2, 0)

class DownSample(nn.Module):
    # downsample block using maxpool
    def __init__(self, c1, c2, n=1, k=1, s=1, g=1):
        super(DownSample, self).__init__()
        dconv = Conv(c1, c2, k, s)
        self.convs = nn.Sequential(dconv, *[Conv(c2, c2, k, s) for _ in range(n-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.convs(x))
    
class UpSample(nn.Module):
    # upsample block using upsample
    def __init__(self, c1, c2, n=1, k=1, s=1, g=1):
        super(UpSample, self).__init__()
        uconv = Conv(c1, c2, k, s)
        self.convs = nn.Sequential(*[Conv(c1, c1, k, s) for _ in range(n-1)], uconv)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.convs(self.upsample(x))

class AEBottleneck(nn.Module):
    # ae bottleneck
    def __init__(self, c1, channels, n=1,  g=1):
        super(AEBottleneck, self).__init__()
        self.encoder = nn.Sequential(
            *[DownSample(cin, cout, n, 3, 1) for (cin, cout) in zip([c1]+channels[:-1], channels)]
        )
        self.decoder = nn.Sequential(
            *[UpSample(cin, cout, n, 3, 1) for (cin, cout) in zip(channels[::-1], channels[-2::-1]+[c1])]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class AEHead(nn.Module):
    # ae reconstruct head
    def __init__(self, c1, channels, n=1, out_ch=3, g=1):
        super(AEHead, self).__init__()
        self.blocks = nn.Sequential(
            *[UpSample(cin, cout, n, 3, 1) for (cin, cout) in zip([c1]+channels[:-1], channels)]
        )
        self.reg = nn.Conv2d(channels[-1], out_ch, 3, 1, 1)
    def forward(self, x):
        return self.reg(self.blocks(x))