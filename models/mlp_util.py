import torch 
import torch.nn as nn 
from einops import rearrange, reduce, repeat

# Transformer encoder layer mainly include a self-attention module and a mlp module, the mlp 
# usually be used to fuse the channels of feature map, for this, we design some scheme in order
# to it.
# 
# input x:  (b, h, w, c) 

# Type 1. MLP Module
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop_prob = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(drop_prob),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        return self.net(x)


# Type 2. Traditional Convolution
class ConvModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        return x


# Type 3. Depthwise Separable Covolution
class DSPConvModule(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        self.net(x) 
        x = x.permute(0, 2, 3, 1)
        return x 


# Type 5. SE Layer
# wish to do  


# Type 4. Channel Fusion
class CFModule(nn.Module):
    def __init__(self, scale_size = (4, 4)) -> None:
        super().__init__()
        self.pool_layer = nn.AdaptiveAvgPool2d(scale_size)
        self.out_act = nn.GELU()
        self.scale = (scale_size[0] * scale_size[1]) ** (-0.5)

    def forward(self, x):
        # x with shape (b, h, w, c)
        identity = x 
        x = x.permute(0, 3, 1, 2)
        x = self.pool_layer(x)
        x = rearrange(x, 'b c h1 w1 -> b c (h1 w1)')
        dots = torch.einsum('bin, bjn -> bij', x, x) * self.scale    # (b, c, c)
        attn = dots.softmax(dim = -1)
        x = torch.einsum('bhwc, btc -> bhwt', identity, attn)
        x = self.out_act(x)
        return x 


# Type 5. Channel Fusion with self-attention
class CFTModule(nn.Module):
    def __init__(self, scale_size = (4, 4)) -> None:
        super().__init__()
        self.dim = scale_size[0] * scale_size[1]
        self.to_qkv = nn.Linear(self.dim, self.dim * 2, bias=False)
        self.pool_layer = nn.AdaptiveAvgPool2d(scale_size)
        self.out_act = nn.GELU()
        self.scale = (self.dim) ** (-0.5)

    def forward(self, x):
        # x with shape (b, h, w, c)
        identity = x 
        x = x.permute(0, 3, 1, 2)
        x = self.pool_layer(x)
        x = rearrange(x, 'b c h1 w1 -> b c (h1 w1)')
        qk = self.to_qkv(x).chunk(2, dim = -1)
        q, k = map(lambda t: t, qk)                 # (b, c, n)

        dots = torch.einsum('bin, bjn -> bij', q, k) * self.scale    # (b, c, c)
        attn = dots.softmax(dim = -1)
        x = torch.einsum('bhwc, btc -> bhwt', identity, attn)   # (b, h, w, c)
        x = self.out_act(x)
        return x 

# Le MLP Module
class LeFF(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer = nn.GELU):
        super().__init__()
        self.fir_lin_layer = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dep_wis_layer = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, padding=1),
                                                     act_layer())
        self.lat_lin_layer = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x):
        x = self.fir_lin_layer(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dep_wis_layer(x)
        x = x.permute(0, 2, 3, 1)
        x = self.lat_lin_layer(x)
        return x


# mlp interface
class MLPZoo(nn.Module):
    def __init__(self, dim, hidden_dim, mlp_type = 'CFT') -> None:
        super().__init__()
        if mlp_type == 'MLP':
            self.mlp_stage = MLP(dim, hidden_dim, drop_prob = 0.)
        elif mlp_type == 'CONV':
            self.mlp_stage = ConvModule(dim) 
        elif mlp_type == 'DSPCONV':
            self.mlp_stage = DSPConvModule(dim)
        elif mlp_type == 'CF':
            self.mlp_stage = CFModule()
        elif mlp_type == 'CFT':
            self.mlp_stage = CFTModule()
        elif mlp_type == 'LEFF':
            self.mlp_stage = LeFF(dim, hidden_dim)
        else:
            raise Exception("mlp type error.")

    def forward(self, x):
        # x with shape (b, h, w, c)
        return self.mlp_stage(x)


if __name__ == "__main__":
    img = torch.rand(1, 16, 16, 5)
    net = MLPZoo(dim = 5, hidden_dim = 10, mlp_type = 'LEFF')
    out = net(img)

    print(img.shape)
    print(out.shape)