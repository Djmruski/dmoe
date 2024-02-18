# From NVIDIA Deep Learning Examples for Tensor Cores
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/utils/metric_logger.py
# Accessed 15-02-2024

from collections import defaultdict, deque
import datetime

import torch

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

    def print_log(self, header, batch_index, num_batches, iter_time, data_time):
        eta_seconds = iter_time.global_avg * (num_batches - batch_index)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        log_msg = [
            header,
            '[{0}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}MB')

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        if (batch_index == 0):
            # print("TARGETS", y)
            # print("PREDS", output)
            if torch.cuda.is_available():
                print(log_msg.format(
                    batch_index, num_batches, eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB)
                )
            else:
                print(log_msg.format(
                    batch_index, num_batches, eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time))
                )