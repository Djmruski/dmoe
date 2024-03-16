
import argparse
import pickle
import time
import torch
import yaml

from base_har import get_data
from trainer import Trainer

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--options',                default=[], nargs='*')
    parser.add_argument('--data-set',               default = None)
    parser.add_argument('--data-path',              default = None)
    parser.add_argument('--features',               default = 405)
    parser.add_argument('--embed-dim',              default = 768)
    parser.add_argument('--patch-size',             default = 45)
    parser.add_argument('--num-classes',            default = 19)
    parser.add_argument('--base-increment',         default = 3)
    parser.add_argument('--increment',              default = 2)
    parser.add_argument('--batch-size',             default = 32)
    parser.add_argument('--n-epochs',               default = 20)
    parser.add_argument('--early-stopping',         default = False)
    parser.add_argument('--patience',               default = 30)
    parser.add_argument('--min-delta',              default = 0)
    parser.add_argument('--restore-best-weights',   default = False)
    parser.add_argument('--rehearsal-samples',      default = 50)
    parser.add_argument('--optimiser',              default = 'SGD', choices = ['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--learning-rate',          default = 0.01)
    parser.add_argument('--weight-decay',           default = 0.0001)
    parser.add_argument('--momentum',               default = 0)
    parser.add_argument('--save-model',             default = False)
    parser.add_argument('--save-dir',               default = 'saves')

    return parser


def load_options(args, options):
    varargs = vars(args)

    name = []
    for o in options:
        with open(o) as f:
            new_opts = yaml.safe_load(f)

        for k, v in new_opts.items():
            if k not in varargs:
                raise ValueError(f'Option {k}={v} doesnt exist!')
        varargs.update(new_opts)
        name.append(o.split("/")[-1].replace('.yaml', ''))

    return '_'.join(name)


def main(args):
    data, task_cla, class_order = get_data(args.data_set, args.data_path, args.num_classes, 
                                       args.base_increment, args.increment)

    print("Classes per Task:", task_cla)
    print("Class Order:", class_order)

    walltime_start, processtime_start = time.time(), time.process_time()

    trainer = Trainer(data, task_cla, class_order, args)
    trainer.train()

    walltime_end, processtime_end = time.time(), time.process_time()
    elapsed_walltime = walltime_end - walltime_start
    elapsed_processtime = processtime_end - processtime_start
    print('Execution time:', )
    print(f"CPU time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_processtime))}\tWall time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_walltime))}")
    print(f"CPU time: {elapsed_processtime}\tWall time: {elapsed_walltime}")

    total_parameters = sum(param.numel() for param in trainer.model.parameters())
    print(f'Parameters: {total_parameters}')

    # Save model and rehearsal data if specified
    if args.save_model:
        trainer.rehearsal.save()
        torch.save(trainer.model, '/'.join([args.save_dir, args.data_set, 'dytox.pth']))
        to_save = {
            'test_confusion_matrix': trainer.test_confusion_matrix,
            'data': {'data': data, 'taskcla': task_cla, 'clsorder': class_order},
            'total_parameters': total_parameters,
        }
        pickle.dump(to_save, open('/'.join([args.save_dir, args.data_set, 'results.pkl']), 'wb'))


if "__main__" in __name__:
    parser = get_args_parser()
    args = parser.parse_args()

    if args.options:
        name = load_options(args, args.options)
    
    main(args)