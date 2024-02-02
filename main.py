
import argparse
import torch

from classifier import Classifier
from dytox import DyTox

def main(args):
    num_classes = 2
    model = DyTox(num_classes, dim=405, B=32, C=45, embed_dim=78)
    # model.head = Classifier(45, 2, 2, 2, 1)

    x = torch.randn(32, 9, 45)
    logits = model(x)
    print(logits.shape)


if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)