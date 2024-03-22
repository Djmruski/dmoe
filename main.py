
import argparse
import pickle
import time
import torch
import yaml

from base_har import get_data
from trainer import Trainer

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--options',                     default=[], nargs='*')
    parser.add_argument('--data-set',                    default = None,    type=int)
    parser.add_argument('--data-path',                   default = None,    type=int)
    parser.add_argument('--features',                    default = 405,     type=int)
    parser.add_argument('--embed-dim',                   default = 768,     type=int)
    parser.add_argument('--patch-size',                  default = 45,      type=int)
    parser.add_argument('--num-classes',                 default = 19,      type=int)
    parser.add_argument('--base-increment',              default = 3,       type=int)
    parser.add_argument('--increment',                   default = 2,       type=int)
    parser.add_argument('--batch-size',                  default = 32,      type=int)
    parser.add_argument('--n-epochs',                    default = 20,      type=int)
    parser.add_argument('--early-stopping',              default = False)
    parser.add_argument('--patience',                    default = 30,      type=int)
    parser.add_argument('--min-delta',                   default = 0)
    parser.add_argument('--restore-best-weights',        default = False)
    parser.add_argument('--rehearsal',                   default = 'GaussianDistribution', choices = ['GaussianDistribution', 'GaussianMixtureModel'])
    parser.add_argument('--rehearsal-samples-per-class', default = 10,      type=int)
    parser.add_argument('--optimiser',                   default = 'SGD',                  choices = ['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--learning-rate',               default = 0.01)
    parser.add_argument('--weight-decay',                default = 0.0001)
    parser.add_argument('--momentum',                    default = 0)
    parser.add_argument('--save-model',                  default = False)
    parser.add_argument('--save-dir',                    default = 'saves')

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
    print(f'Total Parameters: {total_parameters}')

    # Save model and rehearsal data if specified
    if args.save_model:
        trainer.rehearsal.save()
        torch.save(trainer.model, '/'.join([args.save_dir, args.data_set, 'dytox.pth']))
        to_save = {
            'data': {'data': data, 'taskcla': task_cla, 'clsorder': class_order},
            'test_confusion_matrix': trainer.test_confusion_matrix,
            'train_accuracy': trainer.train_accuracy,
            'train_loss': trainer.train_loss,
            'val_accuracy': trainer.val_accuracy,
            'val_loss': trainer.val_loss,
            'total_parameters': total_parameters,
            'train_time': trainer.train_time,
            'prediction_time': trainer.prediction_time,
            'rehearsal_task_creation_time': trainer.rehearsal.task_creation_time,
            'rehearsal_class_creation_time': trainer.rehearsal.class_creation_time,
            'rehearsal_task_build_time': trainer.rehearsal.task_build_time,
            'rehearsal_class_build_time': trainer.rehearsal.class_build_time,
            'train_time_wall': trainer.train_time_wall,
            'prediction_time_wall': trainer.prediction_time_wall,
            'rehearsal_task_creation_time_wall': trainer.rehearsal.task_creation_time_wall,
            'rehearsal_class_creation_time_wall': trainer.rehearsal.class_creation_time_wall,
            'rehearsal_task_build_time_wall': trainer.rehearsal.task_build_time_wall,
            'rehearsal_class_build_time_wall': trainer.rehearsal.class_build_time_wall
        }
        pickle.dump(to_save, open('/'.join([args.save_dir, args.data_set, 'results.pkl']), 'wb'))


if "__main__" in __name__:
    parser = get_args_parser()
    args = parser.parse_args()

    if args.options:
        name = load_options(args, args.options)
    
    main(args)