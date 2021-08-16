import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head  # head数量和每个head的维度
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):  # 2,65,1024 batch,patch+cls_token,dim (每个patch相当于一个token)
        b, n, _, h = *x.shape, self.heads
        # 输入x每个token的维度为1024，在注意力中token被映射16个64维的特征（head*dim_head），
        # 最后再把所有head的特征合并为一个（16*1024）的特征，作为每个token的输出
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 2,65,1024 -> 2,65,1024*3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      qkv)  # 2,65,(16*64) -> 2,16,65,64 ,16个head，每个head维度64
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b,16,65,64 @ b,16,64*65 -> b,16,65,65 : q@k.T
        attn = self.attend(dots)  # 注意力 2,16,65,65  16个head，注意力map尺寸65*65，对应token（patch）[i,j]之间的注意力
        # 每个token经过每个head的attention后的输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # atten@v 2,16,65,65 @ 2,16,65,64 -> 2,16,65,64
        out = rearrange(out, 'b h n d -> b n (h d)')  # 合并所有head的输出(16*64) -> 1024 得到每个token当前的特征
        return self.to_out(out)


# inputs: n L C
# output: n L C
class Former(nn.Module):
    def __init__(self, dim, depth=1, heads=2, dim_head=64, mlp_dim=384, dropout=0.):
        super(Former, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
#                  dim_head=64, dropout=0., emb_dropout=0.):
#         super(ViT, self).__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#
#         num_patchs = (image_height // patch_height) * (image_width // patch_width)  # 8*8
#         patch_dim = channels * patch_height * patch_width  # 3*32*32
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.to_patch_embedding = nn.Sequential(
#             # 3*(8*32)*(8*32) -> (64)*(3*32*32)  64个patch,每个patch的维度为3*32*32
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
#             # b*(64)*(3*32*32) -> b*64*1024  64个patch,每个patch的维度为1024，融合patch中的信息
#             nn.Linear(patch_dim, dim),
#         )
#         # 1,64+1,1024 (patch+pos)
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patchs + 1, dim))
#         # 1,1,1024 (cls_token)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, img):  # 2 3 256 256
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape  # n:patch的数量
#
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # b*1*1024 一个cls_token,维度1024
#         x = torch.cat((cls_tokens, x), dim=1)  # b 65 1024 : 64个patch + 1个cls_token 维度1024 （每个token的维度为1024）
#         x += self.pos_embedding[:, :(n + 1)]  # pos: 1,65,1024 ，patch和cls_token 都有一个1024维度的pos
#         x = self.dropout(x)
#
#         x = self.transformer(x)  # 2，65，1024 -> 2，65，1024
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # 取均值或者取cls_token作为特征 2，1024
#         x = self.to_latent(x)
#         return self.mlp_head(x)  # 2，1024 -> 2，1000
#
#
# if __name__ == "__main__":
#     v = ViT(
#         image_size=256,
#         patch_size=32,
#         num_classes=1000,
#         dim=1024,
#         depth=6,
#         heads=8,
#         mlp_dim=2048,
#         dropout=0.1,
#         emb_dropout=0.1
#     )
#
#     img = torch.randn(2, 3, 256, 256)
#     preds = v(img)  # (1, 1000)
#     print(preds.shape)
