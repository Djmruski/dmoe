import copy
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from attention import Attention
from expert import Expert


class DyTox(nn.module):
    
    def __init__(self, num_classes, dim=405, B=32, C=45, embed_dim=78):
        super.__init__()

        self.dim = dim
        self.B = B
        self.N = dim / C
        self.C = C
        self.embed_dim = embed_dim
        self.num_classes_per_task = [num_classes]

        # tab

        self.task_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, dim))])
        self.task_attn = Attention(self.C, self.embed_dim)

        # clf

        in_dim = (self.N+1) * self.embed_dim
        # TODO: why +1???
        out_dim = self.num_classes_per_task[-1] + 1
        self.experts = nn.ModuleList([Expert(input_size=in_dim, output_size=out_dim)])


    def expand_model(self, num_new_classes):
        """Expand model as per the DyTox framework.

        :param num_new_classes: Number of new classes brought by the new task.
        """
        self.num_classes_per_task.append(num_new_classes)

        # tab

        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)

        # clf

        in_dim = (self.N+1) * self.embed_dim
        # TODO: why +1???
        out_dim = self.num_classes_per_task[-1] + 1
        self.experts.append(Expert(input_size=in_dim, output_size=out_dim))


    def forward_features(self, x):
        B, _, _ = x.shape
        token_embeds = []

        for task_token in self.task_tokens:
            # expand so there is a token for every batch
            task_token = task_token.unsqueeze(1).expand(B, -1, -1)
            # forward through task attention block
            token_embed = self.task_attn(torch.cat((task_token, x), dim=1))
            # flatten vector for classifier
            token_embed = torch.flatten(token_embed, -2, -1)
            token_embeds.append(token_embed)

        return token_embeds

    def forward_classifier(self, token_embeds):
        # clf predictions
        logits = []

        # iterate through each clf
        for i, expert in enumerate(self.experts):
            logits.append(expert(token_embeds[i]))

        return logits

    def forward(self, x):
        token_embeds = self.forward_features(x)
        return self.forward_classifier(token_embeds)