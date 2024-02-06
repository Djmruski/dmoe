
import argparse
import torch

from base_har import get_data
from dytox import DyTox

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-set', default = 'dsads')
    parser.add_argument('--data-path', default = '/home/arw27/CS4099/dmoe/har/DSADS/dsads.mat')
    parser.add_argument('--batch-size', default = 32)
    parser.add_argument('--features', default = 405)
    parser.add_argument('--patches', default = 9)
    parser.add_argument('--embed-dim', default = 45)
    parser.add_argument('--num-classes', default = 19)
    parser.add_argument('--base-increment', default = 3)
    parser.add_argument('--increment', default = 2)

    return parser

def main(args):
    data, task_cla, class_order = get_data(args.data_set, args.data_path, args.num_classes, 
                                       args.base_increment, args.increment)

    print("Class Order:", class_order)

    # num_classes = 2
    # model = DyTox(num_classes, dim=405, B=32, C=45, embed_dim=78)

    # x = torch.randn(32, 9, 45)
    # logits = model(x)
    # print(logits.shape)


if "__main__" in __name__:
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)