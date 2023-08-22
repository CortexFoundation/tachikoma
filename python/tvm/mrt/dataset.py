import typing

from os import path

import tvm

from .types import *

class Dataset:
    def next(self) -> typing.Optional[DataLabelT]:
        """ get next data, None if end. """
        raise RuntimeError("Base Dataset Error")

    def reset(self):
        """ reset dataset internal reader status. """
        raise RuntimeError("Base Dataset Error")

    def __len__(self):
        raise RuntimeError("Base Dataset Error")

    def label(self, index):
        return index

    def labels(self, *indexes):
        return [ self.label(i) for i in indexes ]

class ImageNet(Dataset):
    category_name = "imagenet_category.json"

    def __init__(self):
        base_dir = path.join(path.dirname(__file__), "datasets")

        with open(path.join(base_dir, self.category_name)) as f:
            self.synset = eval(f.read())

    def label(self, index):
        return self.synset.get(index, "unknown category")


class MemoryDataset(Dataset):
    def __init__(self, available_dls: typing.List[DataLabelT]):
        self.data = available_dls
        self._max = len(self.data)
        self._index = 0

    def __len__(self):
        return self._max

    def next(self) -> typing.Optional[DataLabelT]:
        if self._index < self._max:
            self._index += 1
            return self.data[self._index-1]
        return None

    def reset(self):
        self._index = 0
