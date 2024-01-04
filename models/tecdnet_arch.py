import math
import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from timm.models.layers import DropPath

# roll feature map half winsize pixel
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

# Residual Block
class Residual(nn.Module):
    def __init__(self, fn, drop_prob = 0.):
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_prob=drop_prob)

    def forward(self, x, **kwargs):
        return self.drop_path(self.fn(x, **kwargs)) + x

# Layer Norm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PostNorm(nn.Module):
    def __init__(self, dim, fn) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn 

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))

# MLP Module
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim 
        self.hidden_dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

    def flops(self, h, w):
        return h*w*self.dim*self.hidden_dim*2

# this part is most ingenious.
def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

# [NOTE] setting relative distance
def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

# Attention Module
class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.dim = dim
        self.window_size = window_size

        self.var = 0.02
        # self.var = nn.Parameter(torch.tensor(0.2))
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size      # window_size = 8, nw_h = 8
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)      
                                            # out: (batch, head, win_num, win_size, head_dim)
        # normalize
        norm_q = torch.norm(q, dim=-1, keepdim=True)
        q = torch.div(q, norm_q)

        norm_k = torch.norm(k, dim=-1, keepdim=True)
        k = torch.div(k, norm_k)

        # dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) - 1
        dots = torch.div(dots, self.var)
                                            # out: (batch, head, win_num, win_size, win_size)

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0].numpy(), self.relative_indices[:, :, 1].numpy()]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask  

        attn = dots.softmax(dim=-1)                                  # out: (batch, head, win_num, win_size, win_size)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)   # out: (batch, head, win_num, win_size, head_dim)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',   
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
                                                                     # out: (batch, height, width, new_ch)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

    def flops(self, h, w):
        flops = 0
        # to_qkv
        flops += h*w*self.dim*3*self.dim
        # norm
        flops += h*w*self.dim*4
        # attn
        flops += h*w*self.dim*self.window_size*self.window_size
        # attn @ v
        flops += h*w*self.dim*self.window_size*self.window_size
        # out
        flops += h*w*self.dim*self.dim
        return flops 

# transformer encoder layer, including attention module + mlp module
class SwinTransformer(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, drop_path=0.1):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)), drop_prob=drop_path)
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)), drop_prob=drop_path)

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

# each stage include a patch merge layer and double swin trasnformer block
class SwinStage2D(nn.Module):
    def __init__(self, sw_in_channels, layers, num_heads, head_dim=32, window_size=8, 
                       relative_pos_embedding=True, shift=[False, True], drop_path=0.1):
        super().__init__()
        self.in_ch = sw_in_channels
        self.window_size = window_size
        self.layer_num = layers
        # need in_channels equal feature map channels
        assert self.layer_num % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinTransformer(dim=sw_in_channels, heads=num_heads, head_dim=head_dim, mlp_dim=sw_in_channels * 2, shifted=shift[0], 
                                window_size=window_size, relative_pos_embedding=relative_pos_embedding, drop_path=drop_path),
                SwinTransformer(dim=sw_in_channels, heads=num_heads, head_dim=head_dim, mlp_dim=sw_in_channels * 2, shifted=shift[1], 
                                window_size=window_size, relative_pos_embedding=relative_pos_embedding, drop_path=drop_path),
            ]))

    def forward(self, x):
        # x with shape (B, C, H, W)) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)        # rearrange (B, C, H, W)

    def flops(self, h, w):
        flops = 0
        # self-attention + layer norm 
        flops += h*w*self.in_ch
        # to_qkv
        flops += h*w*self.in_ch*3*self.in_ch
        # norm
        # flops += h*w*self.in_ch*4
        # attn
        flops += h*w*self.in_ch*self.window_size*self.window_size
        # attn @ v
        flops += h*w*self.in_ch*self.window_size*self.window_size
        # out
        flops += h*w*self.in_ch*self.in_ch

        # mlp + layer norm
        flops += h*w*self.in_ch
        flops += h*w*self.in_ch*self.in_ch*2*2

        flops = flops * self.layer_num
        return flops 

# ================================================ U Structure =============================================== #

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y + self.shortcut(x)
        return y

    def flops(self, h, w):
        return h*w*self.out_ch*3*3*self.in_ch + h*w*self.out_ch*1*1*self.in_ch + h*w*self.out_ch*3*3*self.out_ch

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, layers=2) -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.pool_layer = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.feat_layer = SwinStage2D(out_channels, layers, out_channels//32, shift=[False, True])

    def forward(self, x):
        return self.feat_layer(self.pool_layer(x))

    def flops(self, h, w):
        return self.feat_layer.flops(h/2, w/2) + h/2*w/2*self.out_ch*self.in_ch*4*4


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, up_token = 'upconv') -> None:
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        if up_token == 'deconv':
            self.up_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        self.feat_layer = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_layer(x1)
        xc = torch.cat([x1, x2], dim = 1)
        return self.feat_layer(xc)

    def flops(self, H, W):
        return 2*H*2*W*self.out_ch*3*3*self.in_ch + self.feat_layer.flops(2*H, 2*W)


class UStructure(nn.Module):
    """ Description: UNet with 4 block for encoder and 4 block for decoder. """
    def __init__(self, img_size = 128, 
                       base_channels = 32, 
                       half_block = 4, 
                       encode_stage = EncoderLayer, 
                       decode_stage = DecoderLayer,
                       encode_layers = [2, 2, 2, 2]):     
        super(UStructure, self).__init__()
        self.img_size = img_size
        self.base_ch = base_channels
        # [32, 64, 128, 256, 512]
        self.half_block = half_block
        self.down_channels_list = [ base_channels*(2**num) for num in range(half_block + 1) ]
        self.up_channels_list = list(reversed(self.down_channels_list))
        self.down_img_size_list = [ img_size//(2**num) for num in range(half_block + 1) ]
        self.encode_layers = encode_layers

        # top layer
        self.fir_layer = nn.Sequential(nn.Conv2d(3, base_channels, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True))
        self.lat_layer = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

        # down layer and up layer module list 
        self.down_module_list = nn.ModuleList([])
        self.up_module_list   = nn.ModuleList([])

        for idx in range(self.half_block):
            self.down_module_list.append(encode_stage(self.down_channels_list[idx], self.down_channels_list[idx + 1], self.encode_layers[idx]))
            self.up_module_list.append(decode_stage(self.up_channels_list[idx], self.up_channels_list[idx + 1]))  
        
        # bottleneck
        self.bottleneck_layer = SwinStage2D(self.down_channels_list[-1], 4, self.down_channels_list[-1] // 32, shift=[False, False])     

    def forward(self, x):
        # first layer
        identity = x
        x = self.fir_layer(x)  

        # encoder layer: [(32, 128, 128), (64, 64, 64), (128, 32, 32), (256, 16, 16), (512, 8, 8)]
        encoder_out_list = [x]
        for encoder_layer in self.down_module_list:
            x = encoder_layer(x)
            encoder_out_list.append(x)
        encoder_out_list = list(reversed(encoder_out_list))

        # bottleneck layer
        x = self.bottleneck_layer(x)

        # decoder 
        for idx, decoder_layer in enumerate(self.up_module_list):
            x = decoder_layer(x, encoder_out_list[idx+1])
        
        # last layer 
        out = self.lat_layer(x) + identity
        return out 

    def flops(self):
        flops = 0
        flops += self.img_size*self.img_size*self.base_ch*3*3*3
        flops += self.img_size*self.img_size*3*3*3*self.base_ch

        for (enc, h) in zip(self.down_module_list, self.down_img_size_list):
            flops += enc.flops(h, h)

        img_rev_list = list(reversed(self.down_img_size_list))
        for (dec, h) in zip(self.up_module_list, img_rev_list):
            flops += dec.flops(h, h)

        flops += self.bottleneck_layer.flops(img_rev_list[0], img_rev_list[0])
        return flops


def RBF_TECDNet_T():
    net = UStructure(base_channels = 16, encode_layers = [2, 2, 2, 2])
    return net 

def RBF_TECDNet_S():
    net = UStructure(base_channels = 32, encode_layers = [2, 2, 2, 2])
    return net 

 
if __name__ == "__main__":
    net = UStructure(256, base_channels = 32, encode_layers = [2, 2, 2, 2])
    print(net.flops())

    # first inference
    import time 

    x = torch.rand(1, 3, 128, 128)
    y = net(x)
    print(y.shape)
