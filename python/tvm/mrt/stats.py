from os import path
import enum

import tvm
import numpy as np

from .utils import PY_ROOT
from .types import *

class StatsConfig(enum.Enum):
    NONE    = enum.auto()
    ALL     = enum.auto()

    ACCURACY    = enum.auto()
    """ enable accuracy info in stats, True by default. """
    TIME        = enum.auto()
    """ enable time logger in stats. """
    DL          = enum.auto()
    """ print current DataLabelT's info, this will suppress all other config. """


class Statistics:
    def __init__(self):
        self.stats_info = {}

    def reset(self):
        """ reset statistic status. """
        raise RuntimeError("Accuracy Type Error")

    def merge(self, dl: DataLabelT):
        """ merge model output and update status. """
        raise RuntimeError("Accuracy Type Error")

    def info(self) -> str:
        """ return statistic information. """
        raise RuntimeError("Accuracy Type Error")

    def dl_info(self) -> str:
        """ return current DataLabel information. """
        raise RuntimeError("Accuracy Type Error")


class ClassificationOutput(Statistics):
    def __init__(self, num_classes=None):
        self.num_classes = None
        self.data, self.label = None

    def merge(self, dl: DataLabelT):
        self.data, self.label = dl
        self.argsort = [ np.argsort(d).tolist() for d in data]

        assert len(data.shape) == 2
        self.batch = data.shape[0]
        if self.num_classes is None:
            self.num_classes = data.shape[1]
        else:
            assert self.num_classes == data.shape[1]

    @property
    def top1(self):
        return [ a[-1] for a in self.argsort ]

    @property
    def top1_raw(self):
        return [ self.data[i][b] for i, b in enumerate(self.top1) ]

    @property
    def top5(self):
        return [ a[-5:] for a in self.argsort ]

    @property
    def top5_raw(self):
        return [ [self.data[i][a] for a in b] \
                for i, b in enumerate(self.top5) ]

    def info(self):
        print("=" * 50)
        print("Batch: {}, Class Number: {}".format(
            self.batch, self.num_classes))
        top1, top1_raw = self.top1, self.top1_raw
        top5, top5_raw = self.top5, self.top5_raw
        for i in range(self.batch):
            print("{:5} Top1: {:3} | Raw: {}".format(
                i, top1[i], top1_raw[i]))
            print("{:5} Top5: {} | Raw: {}".format(
                i, top5[i], top5_raw[i]))
            # print("{:5} Top1: {:3} | Top5: {}".format(
            #     i, top1[i], top5[i]))
        print("=" * 50)

