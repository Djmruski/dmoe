import torch
import torch.nn as nn

from attention import Attention
from new_expert import Expert


class DyTox(nn.module):
    
    def __init__(self, num_classes, dim=405, N=9, C=45, embed_dim=78):
        super.__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.task_attn = Attention(C, embed_dim)
        self.task_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, dim))])
        self.num_classes_per_task = [num_classes]
        self.head = nn.ModuleList([
            Expert(self.dim, self.num_classes_per_task[-1] + 1).cuda()
        ])

    def forward_features(self, x):
        B, _, _ = x.shape
        token_embeds = []

        for task_token in self.task_tokens:
            # expand so there is a token for every batch
            task_token.unsqueeze(1)
            task_token = task_token.expand(B, -1, -1)
            # forward through task attention block
            token_embed = self.task_attn(torch.cat((task_token, x), dim=1))
            # flatten vector for classifier
            token_embed = torch.flatten(token_embed, -2, -1)
            token_embeds.append(token_embed)

        return token_embeds

    def forward_classifier(self, token_embed):
        return None

    def forward(self, x):
        token_embeds = self.forward_features(x)
        return self.forward_classifier(token_embeds)