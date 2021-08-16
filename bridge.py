import torch
from torch import nn, einsum
from einops import rearrange


# inputs: x(b c h w) z(b m d)
# output: z(b m d)
class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, c, dropout=0.):
        super(Mobile2Former, self).__init__()
        inner_dim = heads * c
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        # b l c -> b l h*c -> b h l c
        x = x.contiguous().view(b, h * w, c).repeat(1, 1, self.heads)
        x = x.contiguous().view(b, self.heads, h * w, c)
        k, v = x, x
        # b m d -> b m h*c -> b h m c
        q = self.to_q(z).view(b, self.heads, m, c)
        dots = einsum('b h m c, b h l c -> b h m l', q, k) * self.scale
        # b h m l
        attn = self.attend(dots)
        out = einsum('b h m l, b h l c -> b h m c', attn, v)
        out = rearrange(out, 'b h m c -> b m (h c)')
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, c, dropout=0.):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * c
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, c),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x_ = x.contiguous().view(b, h * w, c).repeat(1, 1, self.heads)
        x_ = x_.contiguous().view(b, self.heads, h * w, c)
        q = x_
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)

        dots = einsum('b h l c, b h m c -> b h l m', q, k) * self.scale
        # b h m l
        attn = self.attend(dots)
        out = einsum('b h l m, b h m c -> b h l c', attn, v)
        out = rearrange(out, 'b h l c -> b l (h c)')
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out
