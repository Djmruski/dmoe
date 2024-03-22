import math
import torch.nn as nn

class Attention(nn.Module):

    """
    This Attention class implements the multi-head attention mechanism as described in the paper
    "Attention is All You Need" by Vaswani et al., 2017. It's a core component of the DyTox Task 
    Attention Block (TAB), facilitating the capture of dependencies without regard to their distance 
    in the input or output sequences.
    
    Reference:
    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
    Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).
    """

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
        # q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # queries using task token
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)                  # queries using entire input
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # computing attention scores
        attn_logits = q @ k.transpose(-2, -1)
        attn_logits = attn_logits / math.sqrt(self.dim)
        attention = attn_logits.softmax(dim=-1)
        values = (attention @ v).transpose(1, 2).reshape(B, 1, C)

        return values