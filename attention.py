import math
import torch.nn as nn

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