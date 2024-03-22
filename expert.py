import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):

    """
    Implementation of the Expert model as inspired by the approach described in the unpublished paper:
    Title: "Continual Learning in Sensor-Based Human Activity Recognition with Dynamic Mixture of Experts"
    Authors: Fahrurrozi Rahman, Martin Schiemer, Andrea Rosales Sanabria, Juan Ye
    Note: Under submission
    DOI: 10.2139/ssrn.4630797

    The Expert model consists of a simple feedforward neural network with one hidden layer, designed
    for use within a Dynamic Mixture of Experts framework for continual learning in sensor-based 
    human activity recognition. The implementation here reflects the principles discussed in the
    paper, focusing on simplicity and modularity to facilitate learning across multiple tasks.

    Page: to appear
    """
    
    def __init__(self, input_size=768, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batchnorm = nn.InstanceNorm1d(hidden_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        out = F.relu(self.batchnorm(self.fc1(x)))
        return self.fc2(out)