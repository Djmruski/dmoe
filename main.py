
import argparse

from classifier import Classifier
from dytox import DyTox

def main(args):
    num_classes = 10
    model = DyTox(num_classes, dim=405, B=32, N=9, C=45, embed_dim=78)
    model.head = Classifier() # TODO:

    

if "__main__" in __name__:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)