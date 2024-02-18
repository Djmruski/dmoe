import datetime
import time
import torch

from collections import defaultdict
from collections import deque
from continuum.metrics import Logger
from timm.utils import accuracy
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from base_har import BaseDataset
from dytox import DyTox


class Trainer:
    def __init__(self, data, task_cla, class_order, args):
        self.data = data
        self.task_cla = task_cla
        self.class_order = class_order
        self.n_epochs = args.n_epochs
        self.args = args

        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = None

        self.logger = Logger(list_subsets=['train', 'test'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_loop(self):
        logger = Logger(list_subsets=['train', 'test'])
        val_loaders = []

        for task_id in range(0, len(self.task_cla)):
            self.model = update_dytox(self.model, task_id, self.args)
            self.model.to(self.device)

            if task_id == 0:
                self.optimiser = AdamW(self.model.parameters(), lr = 0.01)
            else:
                self.model.freeze_old_params()

            # dataloader for training and validation data
            train_dataloader = DataLoader(BaseDataset(self.data[task_id]['trn']), batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            val_dataloader = DataLoader(BaseDataset(self.data[task_id]['val']), batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            val_loaders.append(val_dataloader)

            train_loss = []
            train_acc = []

            for epoch in range(self.n_epochs):
                metric_logger = MetricLogger(delimiter="  ")

                start_time = time.time()
                end = time.time()
                iter_time = SmoothedValue()
                data_time = SmoothedValue()

                ypreds, ytrue = [], []
                # set training mode
                self.model.train()

                running_train_loss = [] # store train_loss per epoch
                dataset_len = 0
                pred_correct = 0.0

                for batch_index, (x, y) in enumerate(train_dataloader):
                    data_time.update(time.time() - end)

                    header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
                    log_msg = [
                        header,
                        '[{0}/{1}]',
                        'eta: {eta}',
                        '{meters}',
                        'time: {time}',
                        'data: {data}'
                    ]

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

                    if torch.cuda.is_available():
                        log_msg.append('max mem: {memory:.0f}')
                    log_msg = metric_logger.delimiter.join(log_msg)
                    eta_seconds = iter_time.global_avg * (len(train_dataloader) - batch_index)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    MB = 1024.0 * 1024.0

                    if (batch_index == 0):
                        # print("TARGETS", y)
                        # print("PREDS", output)
                        if torch.cuda.is_available():
                            print(log_msg.format(
                                batch_index, len(train_dataloader), eta=eta_string,
                                meters=str(metric_logger),
                                time=str(iter_time), data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / MB)
                            )
                        else:
                            print(log_msg.format(
                                batch_index, len(train_dataloader), eta=eta_string,
                                meters=str(metric_logger),
                                time=str(iter_time), data=str(data_time))
                            )

                    loss.backward()
                    self.optimiser.step()
                    # zero gradients for new batch 
                    self.optimiser.zero_grad()

            # Validating Model

            self.evaluate(val_loaders, logger)
        torch.save(self.model, 'models/dytox.pth')


    @torch.no_grad()
    def evaluate(self, val_loaders, logger):
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
        criterion = nn.CrossEntropyLoss()

        metric_logger = MetricLogger(delimiter="  ")

        for task_id, val_loader in enumerate(val_loaders):
            start_time = time.time()
            end = time.time()
            iter_time = SmoothedValue(fmt='{avg:.4f}')
            data_time = SmoothedValue(fmt='{avg:.4f}')

            for batch_index, (x, y) in enumerate(val_loader):
                header = 'Test:'
                log_msg = [
                    header,
                    '[{0}/{1}]',
                    'eta: {eta}',
                    '{meters}',
                    'time: {time}',
                    'data: {data}'
                ]

                data_time.update(time.time() - end)

                x = x.to(self.device)
                y = y.type(torch.LongTensor).to(self.device)
                output = self.model(x)

                print("TARGETS", y)
                print("PREDS", output)

                loss = criterion(output, y)

                # print(loss)
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
                task_ids = torch.full_like(predictions, fill_value=task_id)

                logger.add([predictions, targets, task_ids], subset='test')
                
                iter_time.update(time.time() - end)

                # Print Metrics

                if torch.cuda.is_available():
                    log_msg.append('max mem: {memory:.0f}')
                log_msg = metric_logger.delimiter.join(log_msg)
                eta_seconds = iter_time.global_avg * (len(val_loader) - batch_index)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                MB = 1024.0 * 1024.0

                if torch.cuda.is_available():
                    print(log_msg.format(
                        batch_index, len(val_loader), eta=eta_string,
                        meters=str(metric_logger),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                    )
                else:
                    print(log_msg.format(
                        batch_index, len(val_loader), eta=eta_string,
                        meters=str(metric_logger),
                        time=str(iter_time), data=str(data_time))
                    )
                
                end = time.time()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(val_loader)))



def update_dytox(model, task_id, args):
    if task_id == 0:
        print(f'Creating DyTox')
        model = DyTox(args.base_increment, args.features, args.batch_size, 
                      args.patch_size, args.embed_dim)
    else:
        print(f'Expanding model')
        model.expand_model(args.increment)

    return model

# From NVIDIA Deep Learning Examples for Tensor Cores
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/utils/metric_logger.py
# Accessed 15-02-2024

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.series.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1]

    @property
    def global_avg(self):
        return self.total / self.count
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def get_dict(self):
        loss_dict = {}
        for name, meter in self.meters.items():
            loss_dict[name] = "{:.4f} ({:.4f})".format(meter.median, meter.global_avg)
        return loss_dict