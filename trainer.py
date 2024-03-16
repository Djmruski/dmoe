import datetime
import time
import numpy as np
import torch

import rehearsal

from continuum.metrics import Logger
from sklearn.metrics import classification_report, f1_score
from timm.utils import accuracy
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from base_har import BaseDataset
from earlystopping import EarlyStopping
from dytox import DyTox
from logger import SmoothedValue, MetricLogger


class Trainer:

    """
    A class to handle training of the DyTox model for continual learning tasks.
    
    Attributes:
        data (dict): Dataset for training, structured by tasks.
        task_cla (list): List containing the number of classes per task.
        class_order (list): The order of classes across all tasks.
        n_epochs (int): Number of epochs to train each task.
        args (Namespace): Command line arguments specifying model and training configurations.
        model (DyTox): Instance of the DyTox model to be trained.
        rehearsal (Rehearsal): Rehearsal mechanism for generating pseudo-data.
        criterion (nn.Module): Loss function for training.
        optimiser (Optimizer): Optimizer for training the model.
        logger (Logger): Logger for recording training and evaluation metrics.
        device (str): Device to run the training on ('cuda' or 'cpu').
    """

    def __init__(self, data, task_cla, class_order, args):
        self.data = data
        self.task_cla = task_cla
        self.class_order = class_order
        self.n_epochs = args.n_epochs
        self.early_stop = args.early_stopping
        self.patience = args.patience
        self.min_delta = args.min_delta
        self.restore_best_weights = args.restore_best_weights
        self.args = args

        print(f'Creating DyTox')
        self.model = DyTox(args.base_increment, args.features, args.embed_dim, args.patch_size)
        rehearsal_class_ = getattr(rehearsal, args.rehearsal)
        self.rehearsal = rehearsal_class_(args.data_set, args.rehearsal_samples_per_class, path=args.save_dir)
        self.criterion = nn.CrossEntropyLoss()

        optimisers = {
            'SGD': optim.SGD(self.model.parameters(), lr=args.learning_rate, 
                                                    momentum=args.momentum, 
                                                    weight_decay=args.weight_decay),
            'Adam': optim.Adam(self.model.parameters(), lr=args.learning_rate,
                                                    weight_decay=args.weight_decay),
            'AdamW': optim.AdamW(self.model.parameters(), lr=args.learning_rate,
                                                    weight_decay=args.weight_decay)
        }

        self.optimiser = optimisers[args.optimiser]

        self.logger = Logger(list_subsets=['train', 'test'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        """logging properties"""
        self.test_confusion_matrix = {} # format: {'true': [], 'preds': [], 'labels': [], 0: {'true': [], 'preds': [], 'labels': []}}
        # self.train_accuracy = {}        # format: {0: 0.5, 1: 0.5, ...}
        # self.train_loss = {}            # format: {0: 0.5}, 1: 0.5, ...}
        # self.val_accuracy = {}          # format: {0: 0.5, 1: 0.5, ...}
        # self.val_loss = {}              # format: {0: {expert: 0.5, gate: 0.5}, 1: {expert: 0.5, gate: 0.5}, ...}
        # self.expert_train_time = {}
        # self.expert_train_time_wall = {} 
        # self.gate_train_time = {}
        # self.gate_train_time_wall = {}
        # self.prediction_time = None     # format: floating nano seconds
        # self.prediction_time_wall = None
        # self.task_params = {}


    def train(self):
        """
        Main training loop for the DyTox model across all tasks. Handles model expansion,
        rehearsal data generation and integration, and model evaluation.
        """
        logger = Logger(list_subsets=['train', 'test'])
        val_loaders = []

        for task_id in range(len(self.task_cla)):

            early_stopping = EarlyStopping(self.patience, self.min_delta, 
                                           self.restore_best_weights) if self.early_stop else None

            if task_id == 0:
                self.rehearsal.add_task(self.data[task_id]['trn'])
            else:   # For all subsequent tasks
                print(f'Expanding model')
                self.model.expand_model(self.args.increment)
                self.model.freeze_old_params()

                # Generate and integrate rehearsal data
                task_data = self.data[task_id]['trn']['x']
                task_labels = self.data[task_id]['trn']['y']
                rehearsal_data, rehearsal_labels = self.rehearsal.generate_rehearsal_data()
                augmented_data = np.concatenate([task_data, rehearsal_data])
                augmented_labels = np.concatenate([task_labels, rehearsal_labels])

                # Update the dataset with augmented data
                self.rehearsal.add_task(self.data[task_id]['trn'])
                self.data[task_id]['trn']['x'] = augmented_data
                self.data[task_id]['trn']['y'] = augmented_labels

            self.model.to(self.device)

            # print(len(self.data[task_id]['val']['y']))

            # Prepare data loaders
            train_dataloader = DataLoader(BaseDataset(self.data[task_id]['trn']), 
                                          batch_size=self.args.batch_size, shuffle=True)
            val_dataloader = DataLoader(BaseDataset(self.data[task_id]['val']), 
                                        batch_size=self.args.batch_size, shuffle=True)
            val_loaders.append(val_dataloader)

            for epoch in range(self.n_epochs):
                self.train_one_epoch(task_id, epoch, train_dataloader, early_stopping)
                if self.early_stop and early_stopping.stop:
                    print(f"Early stopping: exit epoch {epoch+1}.")
                    break

            all_true, all_preds = self.evaluate(val_loaders, logger)
        
        self.test_confusion_matrix['true'] = all_true
        self.test_confusion_matrix['preds'] = all_preds
        self.test_confusion_matrix['labels'] = sorted(np.unique(all_true))

        print("\n====================\n")
        print(f"f1_score(micro): {100 * f1_score(all_true, all_preds, average='micro', zero_division=1)}")
        print(f"f1_score(macro): {100 * f1_score(all_true, all_preds, average='macro', zero_division=1)}")
        print(f"Classification report:\n{classification_report(all_true, all_preds, zero_division=1)}")


    def train_one_epoch(self, task_id, epoch, data_loader, early_stopping):
        """
        Trains the model for one epoch on a given task's training data.
        
        Args:
            task_id (int): Current task identifier.
            epoch (int): Current epoch number.
            data_loader (DataLoader): DataLoader for the current task's training data.
        """
        metric_logger = MetricLogger(delimiter="  ")

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue()
        data_time = SmoothedValue()

        self.model.train()  # Set the model to training mode

        for batch_index, (x, y) in enumerate(data_loader):
            data_time.update(time.time() - end)

            x = x.to(self.device)
            y = y.type(torch.LongTensor).to(self.device)
            output = self.model(x)

            loss = self.criterion(output, y)
            acc1, acc5 = accuracy(output, y, topk=(1, min(5, output.shape[1])))

            # Log Metrics
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=x.shape[0])
            metric_logger.meters['acc5'].update(acc5.item(), n=x.shape[0])

            iter_time.update(time.time() - end)

            # Print Metrics
            header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
            metric_logger.print_log(header, batch_index, len(data_loader), iter_time, data_time)

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()  # Zero gradients for the next batch

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(data_loader)))
        
        if self.early_stop:
            early_stopping(loss, self.model)


    @torch.no_grad()
    def evaluate(self, data_loader, logger):
        """
        Evaluates the model on validation data for all tasks seen so far.

        Args:
            current_task_id (int): Identifier of the current task being evaluated.
            data_loaders (list): List of DataLoader objects for validation data of each task.
            logger (Logger): Logger for recording evaluation metrics.
        """
        metric_logger = MetricLogger(delimiter="  ")
        self.model.eval()  # Set the model to evaluation mode

        all_preds, all_true = [], []
        for task_id, val_loader in enumerate(data_loader):
            ypreds, ytrue = [], []

            start_time = time.time()
            end = time.time()
            iter_time = SmoothedValue(fmt='{avg:.4f}')
            data_time = SmoothedValue(fmt='{avg:.4f}')

            for batch_index, (x, y) in enumerate(val_loader):
                data_time.update(time.time() - end)

                x = x.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                output = self.model(x)

                loss = self.criterion(output, y)
                acc1, acc5 = accuracy(output, y, topk=(1, min(5, output.shape[1])))

                # Log Metrics
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=x.shape[0])
                metric_logger.meters['acc5'].update(acc5.item(), n=x.shape[0])

                # Convert task_id to a tensor and expand its dimensions to match predictions and 
                # targets. Assuming task_id is a scalar, use torch.full to create a tensor of the 
                # same shape as predictions filled with the task_id value
                predictions = output.detach().cpu().argmax(dim=1)
                targets = y.detach().cpu()
                task_ids = torch.full_like(predictions, task_id)
                logger.add([predictions, targets, task_ids], subset='test')
                ypreds.extend(predictions)
                ytrue.extend(targets)
                all_preds.extend(predictions)
                all_true.extend(targets)

                iter_time.update(time.time() - end)

                # Print Metrics
                header = 'Test:'
                metric_logger.print_log(header, batch_index, len(data_loader), iter_time, data_time)

                end = time.time()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, 
                                                             total_time / len(val_loader)))
        
        return all_true, all_preds