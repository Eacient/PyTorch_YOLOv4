import torch.nn.functional as F

import sys
sys.path.append('.')
from models.common import *
from timm.layers import trunc_normal_


from timm.layers import to_2tuple, DropPath

def channel_shuffle_conv(x, g):
    batchsize, num_channels, L = x.data.size()
    assert num_channels % g == 0
    group_channels = num_channels // g
    
    x = x.reshape(batchsize, group_channels, g, L)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(batchsize, num_channels, L)
    return x

def channel_shuffle(x, g):
    batchsize, L, num_channels = x.data.size()
    assert num_channels % g == 0
    group_channels = num_channels // g
    
    x = x.reshape(batchsize, L, group_channels, g)
    x = x.permute(0, 1, 3, 2)
    x = x.reshape(batchsize, L, num_channels)
    return x

class Mlp(nn.Module):
    def __init__(self, c1, c2, e=4, drop=0.):
        super().__init__()
        c_ = int(c2 * e)
        self.fc1 = nn.Linear(c1, c_)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c_, c2)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GroupLinear(nn.Module):
    def __init__(self, c1, c2, g=1):
        super().__init__()
        assert c1 % g == 0 and c2 % g == 0
        self.c1=c1
        self.c2=c2
        self.g = g
        self.weight = nn.Parameter(torch.zeros(g, c1//g, c2//g), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(g, 1, c2//g), requires_grad=True)
    
    def forward(self, x):
        b, N, g, c1, c2 = *x.shape[:1], self.g, self.c1, self.c2
        return torch.bmm(x.reshape(b*N, g, c1//g).transpose(0,1), self.weight).transpose(0,1).reshape(b, N, g, c2//g)

class ShuffleMlp(nn.Module):
    def __init__(self, c1, c2, e=4, g=4, drop=0.):
        super().__init__()
        c_ = int(c2 * e)
        self.c1 = c1
        self.c_ = c_
        self.c2 = c2
        self.g = g
        # self.fc1 = nn.Conv1d(c1, c_, 1, 1, 0, groups=g)
        self.weight1 = nn.Parameter(torch.zeros(g, c1//g, c_//g), requires_grad=True)
        self.act = nn.GELU()
        # self.fc2 = nn.Conv1d(c_, c2, 1, 1, 0, groups=g)
        self.weight2 = nn.Parameter(torch.zeros(g, c_//g, c2//g), requires_grad=True)
        self.drop = nn.Dropout(drop)
        trunc_normal_(self.weight1, std=0.2)
        trunc_normal_(self.weight2, std=0.2)
        
    def forward(self, x):
        b, N, g, c1, c_, c2 = *x.shape[:2], self.g, self.c1, self.c_, self.c2
        x = torch.bmm(x.reshape(b*N, g, c1//g).transpose(0,1), self.weight1).transpose(0,1)
        x = self.act(x)
        x = x.transpose(1,2).reshape(b*N, g, c_//g)
        x = self.drop(x)
        x = torch.bmm(x.transpose(0,1), self.weight2).transpose(0,1)
        x = self.drop(x)
        x = x.reshape(b, N, c2)
        return x
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_relative_coords_table(window_size, pretrained_window_size):
    # get relative_coords_table
    relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
    relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
    relative_coords_table = torch.stack(
        torch.meshgrid([relative_coords_h,
                        relative_coords_w], indexing="ij")).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
    if pretrained_window_size[0] > 0:
        relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
    else:
        relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)
    relative_coords_table *= 8  # normalize to -8, 8
    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / np.log2(8)
    
    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

    return relative_coords_table, relative_position_index

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_table, relative_position_index = get_relative_coords_table(self.window_size, pretrained_window_size)

        self.register_buffer("relative_coords_table", relative_coords_table)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp() # attn temp
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

def calculate_sw_mask(shift_size, input_resolution, window_size):
    if shift_size <= 0:
        return None
    
    # calculate attention mask for SW-MSA
    H, W = input_resolution
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, c1, c2, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., g=1, qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                 pretrained_window_size=0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.g = g
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(c1)
        self.attn = WindowAttention(
            c1, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(c2)
        self.mlp = ShuffleMlp(c1, c2, mlp_ratio, g, drop=drop) if g > 1 else Mlp(c1, c2,  mlp_ratio, drop=drop)

        attn_mask = calculate_sw_mask(self.shift_size, self.input_resolution, self.window_size)
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.c1, self.c2}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.c1 * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += H * W * (self.c1+self.c2) * self.c2 * self.mlp_ratio / self.g
        # norm2
        flops += self.c2 * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution=14):
        super().__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops

class SwinLayer(nn.Module):
    def __init__(self, c1, c2, n=1, num_heads=12, down_sample=False, g=1,
                 window_spec = (14, 7, 0), drop_spec=(0., 0. , 0.),
                 mlp_ratio=4., qkv_bias=True):
        super().__init__()
        input_resolution, window_size, pretrained_window_size = window_spec
        drop, attn_drop, drop_path = drop_spec 
        self.c1 = c1
        self.c2 = c2
        self.input_resolution = to_2tuple(input_resolution)
        self.depth = n
        
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                c1, c2, input_resolution, num_heads, window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                pretrained_window_size=pretrained_window_size,
                mlp_ratio=mlp_ratio, g=g,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
            ) for i in range(n)
        ])
        self.down_sample = PatchMerging(c2, input_resolution) if down_sample else nn.Identity()
        
    def forward(self, x):
        x = self.blocks(x)
        x = self.down_sample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.c1, self.c2}, input_resolution={self.input_resolution}, depth={self.depth}"
    
    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if type(self.down_sample) is not nn.Identity:
            flops += self.down_sample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, img_size=224, patch_size=4, norm_layer=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class SegPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.pool(x)
        x = x.flatten(1)
        return x

if __name__ == "__main__":
    # mlp = Mlp(192, 192, 4)
    # smlp = ShuffleMlp(192, 192, 4, 4)
    # layer = SwinLayer(192, 192, 2, 12, 4, 4)
    # mlp(torch.ones(1, 196, 192))
    # smlp(torch.ones(1, 196, 192))
    # layer(torch.ones(1, 196, 192))
    c1 = 192
    c2 = 192
    
    # nn.Linear()
    # l1 = GroupLinear(c1, c2, 1)
    # l2 = GroupLinear

