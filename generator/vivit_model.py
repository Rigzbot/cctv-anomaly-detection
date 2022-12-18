import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from generator.patch_expander_layer import PatchExpand
from generator.basic_upsampling_layer import BasicLayer_up
from generator.vivit_util import PreNorm, Attention, FeedForward

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim=768, depth=4, heads=3, pool='cls',
                 in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        np = image_size // (patch_size ** 2)
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, np, num_frames + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # def forward(self, x):
    #     print(x.shape, 'original shape in forward block')
    #     x = self.to_patch_embedding(x)
    #     print(x.shape, 'after patch embedding and dimensionality reduction(before concatination)')
    #     b, t, n, _ = x.shape
    #     cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
    #     x = torch.cat((cls_space_tokens, x), dim=2)
    #     x += self.pos_embedding[:, :, :(n + 1)]
    #     x = self.dropout(x)
    #     print(x.shape, 'after concat and pos embedding')
    #     x = rearrange(x, 'b t n d -> (b t) n d')
    #     print(x.shape, 'before spatial transformer')
    #     x = self.space_transformer(x)
    #     print(x.shape, 'after spatial transformer')
    #     x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
    #     print(x.shape, 'before temporal embdding')
    #     cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
    #     print(x.shape, 'after temporal embedding')
    #     x = torch.cat((cls_temporal_tokens, x), dim=1)
    #     print(x.shape, 'before temporal transformer')
    #     x = self.temporal_transformer(x)
    #     print(x.shape, 'after temporal transformer')
    #     x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
    #     print(x.shape, 'final shape')

    #     return self.mlp_head(x)

    def forward(self, x):
        # print(x.shape, 'original shape in forward block')
        x = self.to_patch_embedding(x)
        # print(x.shape, 'after patch embedding and dimensionality reduction(before concatination)')
        x = x.permute((0, 2, 1, 3))  # b t n d -> b n t d
        b, n, t, _ = x.shape
        cls_space_tokens = repeat(self.space_token, '() t d -> b n t d', b=b, n=n)
        x = torch.cat((cls_space_tokens, x), dim=2)
        # print(x.shape)
        x += self.pos_embedding2[:, :, :(t + 1)]
        # x += torch.randn(1, 16, 5, 512)
        x = self.dropout(x)
        # print(x.shape, 'after concat and pos embedding')
        x = rearrange(x, 'b n t d -> (b n) t d')
        # print(x.shape, 'before temporal transformer')
        x = self.temporal_transformer(x)
        # print(x.shape, 'After temporal transformer')
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        # x += self.pos_embedding[:, :, :]
        # print(x.shape, 'before spatial transformer')
        x = self.space_transformer(x)
        # print(x.shape, 'after spatial transformer')
        expand = PatchExpand((4, 4), 768 * 2, dim_scale=2)
        upsample = BasicLayer_up(768 * 2, (4, 4), 2, 2, 2, upsample=expand)
        x = upsample(x)
        # print(x.shape, "final shape before upsample")
        return x


if __name__ == "__main__":
    n_frames = 4
    img = torch.ones([16, n_frames, 768, 8, 8])

    model = ViViT(64, 2, 2, n_frames, dim=768 * 2, in_channels=768)
    out = model(img)
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params / 1e6)
    print(out.shape)

    # print("Shape of out :", out.shape)      # [B, num_classes]