import datetime
import pickle
import time
import torch

from continuum.metrics import Logger
from timm.utils import accuracy
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from base_har import BaseDataset
from dytox import DyTox
from logger import SmoothedValue, MetricLogger
from rehearsal import Rehearsal


class Trainer:
    def __init__(self, data, task_cla, class_order, args):
        self.data = data
        self.task_cla = task_cla
        self.class_order = class_order
        self.n_epochs = args.n_epochs
        self.args = args

        self.model = None
        self.rehearsal = Rehearsal()
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = None

        self.logger = Logger(list_subsets=['train', 'test'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def train(self):
        logger = Logger(list_subsets=['train', 'test'])
        val_loaders = []

        for task_id in range(len(self.task_cla)):
            self.model = update_dytox(self.model, task_id, self.args)
            self.model.to(self.device)

            task_data = self.data[task_id]['trn']

            # For the first task, the optimiser must be initialised
            # Rehearsal data is not required
            if task_id == 0:
                self.optimiser = AdamW(self.model.parameters(), lr = 0.01)
            # For following tasks, old tokens and experts must be frozen
            # Rehearsal data must be added to the data loader
            else:
                self.model.freeze_old_params()

            # dataloader for training and validation data
            train_dataloader = DataLoader(BaseDataset(self.data[task_id]['trn']), batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            val_dataloader = DataLoader(BaseDataset(self.data[task_id]['val']), batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            val_loaders.append(val_dataloader)

            for epoch in range(self.n_epochs):
                self.train_one_epoch(task_id, epoch, train_dataloader)

            self.evaluate(task_id, val_loaders, logger)

        # Save Model

        torch.save(self.model, 'models/dytox.pth')


    def train_one_epoch(self, task_id, epoch, data_loader):
        metric_logger = MetricLogger(delimiter="  ")

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue()
        data_time = SmoothedValue()

        # set training mode
        self.model.train()

        for batch_index, (x, y) in enumerate(data_loader):
            data_time.update(time.time() - end)

            x = x.to(self.device)
            y = y.type(torch.LongTensor).to(self.device)

            output = self.model(x, False)

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
            self.optimiser.zero_grad()  # zero gradients for new batch 
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(data_loader)))


    @torch.no_grad()
    def evaluate(self, current_task_id, data_loader, logger):
        """
        Top-1 accuracy is the standard accuracy metric, indicating the percentage of times the model's 
        highest-confidence prediction (i.e., the top prediction) matches the true label of the input. 

        Top-5 accuracy is more specific to scenarios where the model provides a ranked list of 
        predictions. It measures the percentage of test samples for which the true label is among the 
        model's top 5 predictions. This metric is particularly useful for evaluating models in tasks 
        where there are many possible categories (such as in large-scale image classification problems) 
        and where being "close" to the correct answer is still valuable. Top-5 accuracy is a more 
        lenient metric than Top-1, as it allows for a correct prediction to be anywhere in the top 5 
        ranked predictions made by the model.
        """
        save = False
        metric_logger = MetricLogger(delimiter="  ")

        for task_id, val_loader in enumerate(data_loader):
            start_time = time.time()
            end = time.time()
            iter_time = SmoothedValue(fmt='{avg:.4f}')
            data_time = SmoothedValue(fmt='{avg:.4f}')

            for batch_index, (x, y) in enumerate(val_loader):
                # if current_task_id == task_id and batch_index == len(val_loader) - 1:
                #     with open('models/embeddings/e{}_preds.pkl'.format(task_id), 'wb') as file:
                #         pickle.dump(y, file)
                #     save = True
                data_time.update(time.time() - end)

                x = x.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                output = self.model(x, save)

                loss = self.criterion(output, y)
                acc1, acc5 = accuracy(output, y, topk=(1, min(5, output.shape[1])))

                # Log Metrics

                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=x.shape[0])
                metric_logger.meters['acc5'].update(acc5.item(), n=x.shape[0])

                # Convert task_id to a tensor and expand its dimensions to match predictions and targets
                # Assuming task_id is a scalar, use torch.full to create a tensor of the same shape as predictions
                # filled with the task_id value
                predictions = output.cpu().argmax(dim=1)
                targets = y.cpu()
                task_ids = torch.full_like(predictions, task_id)
                logger.add([predictions, targets, task_ids], subset='test')

                iter_time.update(time.time() - end)

                # Print Metrics

                header = 'Test:'
                metric_logger.print_log(header, batch_index, len(data_loader), iter_time, data_time)

                end = time.time()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, 
                                                             total_time / len(val_loader)))


def update_dytox(model, task_id, args):
    if task_id == 0:
        print(f'Creating DyTox')
        model = DyTox(args.base_increment, args.features, args.batch_size, 
                      args.patch_size, args.embed_dim)
    else:
        print(f'Expanding model')
        model.expand_model(args.increment)

    return model