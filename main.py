
import argparse
import torch

from classifier import Classifier
from dytox import DyTox

def main(args):
    num_classes = 10
    model = DyTox(num_classes, dim=405, B=32, C=45, embed_dim=78)
    # model.head = Classifier() # TODO:

    x = torch.randn(32, 9, 45)
    out = model(x)
    print(out[0].shape)


if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)