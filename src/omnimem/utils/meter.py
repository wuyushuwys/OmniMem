from collections import deque
from timeit import default_timer
import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    w/ moving average support
    """

    def __init__(self, max_length=1000):
        self._val_queue = deque([], maxlen=max_length)
        self._count_queue = deque([], maxlen=max_length)
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    @torch.no_grad()
    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.item()
        self._val = val
        self._sum += val * n
        self._count += n
        self._val_queue.append(val * n)
        self._count_queue.append(n)
        self._avg = self._sum / self._count

    @property
    def val(self):
        return self._val

    @property
    def mavg(self):
        try:
            return sum(self._val_queue) / sum(self._count_queue)
        except ZeroDivisionError:
            return 0

    @property
    def avg(self):
        return self._avg


class TimerMeter(AverageMeter):

    def __init__(self, max_length=1000, wait_for_all=False):
        super(TimerMeter, self).__init__(max_length=max_length)
        self.wait_for_all = wait_for_all
        self.start = default_timer()

    def tic(self):
        self.start = default_timer()

    def toc(self):
        if self.wait_for_all and torch.distributed.is_initialized():
            torch.distributed.barrier()
        elapsed = default_timer() - self.start
        self.update(elapsed)


class LossesMeter:
    """Computes and stores the average and current value"""

    def __init__(self, max_length=1000):
        self._loss = {}
        self.max_length = max_length

    def update(self, loss_dict: dict, size=1):
        for key, value in loss_dict.items():
            value = value.item() if torch.is_tensor(value) else value
            if key not in self._loss.keys():
                self._loss[key] = AverageMeter(max_length=self.max_length)
                self._loss[key].update(value, n=size)
            else:
                self._loss[key].update(value, n=size)

    def reset(self):
        self._loss = {}

    @property
    def avg(self):
        return {f"{key}": meter.avg for key, meter in self._loss.items()}

    @property
    def mavg(self):
        return {f"{key}": meter.mavg for key, meter in self._loss.items()}

    @property
    def val(self):
        return {key: meter.val for key, meter in self._loss.items()}

    def getter(self, name, type='avg'):
        return getattr(self._loss[name], type)
