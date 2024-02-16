import math
import numpy as np
import torch
import torch.nn as nn


# Make multi-head with self.heads = 1
class Attention(nn.Module):
    def __init__(self, dim, embed_dim, num_heads=1):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q = nn.Linear(self.dim, self.dim)
        self.k = nn.Linear(self.dim, self.dim)
        self.v = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        B, N, C = x.shape

        # creating queries, keys, vectors using input vectors
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # computing attention scores
        attn_logits = q @ k.transpose(-2, -1)
        attn_logits = attn_logits / math.sqrt(self.dim)
        attention = attn_logits.softmax(dim=-1)
        values = (attention @ v).transpose(1, 2).reshape(B, N, C)

        return values

# batch_size = 32
# dim = [1, 405]
# dim_alt = [9, 45]

# x = torch.randn(batch_size, dim_alt[0], dim_alt[1])
# task_token = torch.randn(batch_size, 1, dim_alt[1])
# xx = torch.cat((task_token, x), dim=1)
# attn = Attention(45, 78)
# token_embed = attn(xx)
# token_embed = torch.flatten(token_embed, -2, -1)