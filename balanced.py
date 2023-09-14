import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self):
        super().__init__()
        # self.sample_per_class = torch.tensor(freq)

    def forward(self, logits, label, freq, reduction='mean'):
        # "freq" is given as they can change every time it is called
        return balanced_softmax_loss(label, logits, torch.tensor(freq), reduction)
        # return balanced_softmax_loss(label, logits, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    print(f"sample_per_class: {sample_per_class}")
    spc = sample_per_class.type_as(logits)    
    temp_spc = spc.unsqueeze(0).expand(logits.shape[0], -1)    
    try:
      logits = logits + temp_spc.log()
    except:
      print(f"logits.size(): {logits.size()}")
      print(f"spc.size(): {sample_per_class.size()}")
      print(f"temp_spc.size(): {temp_spc.size()}")
      exit()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def create_loss():
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax()