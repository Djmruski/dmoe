import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

class Expert(nn.Module):
    """
    The Expert model. It has only one hidden layer with hidden_size units.
    """
    
    def __init__(self, input_size=768, hidden_size=20, output_size=2, projected_output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.mapper = nn.Linear(in_features=output_size, out_features=projected_output_size, bias=False)
        self.batchnorm = nn.InstanceNorm1d(num_features=hidden_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = F.relu(self.batchnorm(self.fc1(x)))
        out = self.mapper(self.fc2(out))
        return out

class BiasLayer(torch.nn.Module):
    """
    The bias layer adapted from BiC.
    It will be added to the end of expert classification layers.
    It has only two parameters: alpha and beta
    """
    
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return self.alpha * x + self.beta        