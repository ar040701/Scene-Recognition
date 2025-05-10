import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)  # [B, num_patches, emb_size]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.positions

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        B, N, E = x.shape
        H = self.num_heads
        q = self.queries(x).reshape(B, N, H, E // H).transpose(1, 2)
        k = self.keys(x).reshape(B, N, H, E // H).transpose(1, 2)
        v = self.values(x).reshape(B, N, H, E // H).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(E // H)
        attn = self.att_drop(scores.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, E)
        return self.projection(out)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ViT(nn.Module):
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=40,
                 emb_size=768,
                 depth=12,
                 num_heads=12,
                 drop_p=0.1,
                 forward_expansion=4,
                 forward_drop_p=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.encoder = TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            drop_p=drop_p,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_drop_p,
            num_heads=num_heads
        )
        self.head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.head(x[:, 0])