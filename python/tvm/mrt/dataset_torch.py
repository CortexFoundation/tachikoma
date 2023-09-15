import typing
from os import path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision as tv

from .types import DataLabelT
from . import dataset, utils

class TorchWrapperDataset(dataset.Dataset):
    def __init__(self, data_loader: DataLoader):
        self._loader = data_loader
        self._iter = iter(self._loader)
        self._len = len(self._loader)

    def reset(self):
        self._iter = iter(self._loader)

    def resize(self, batch_size: int) -> dataset.Dataset:
        return TorchWrapperDataset(DataLoader(
            self._loader.dataset,
            batch_size=batch_size))

    def __len__(self):
        return self._len

    def next(self) -> typing.Optional[DataLabelT]:
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            #  raise e
            print("error:", e)
            return None, None

class TorchImageNet(dataset.ImageNet):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.ImageFolder(
                path.join(utils.MRT_DATASET_ROOT, "imagenet/val"),
                transform=self._to_tensor)
        self.data_loader = DataLoader(
                val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def _to_tensor(self, img: Image.Image):
        img = img.resize(self._img_size)
        img = np.array(img).astype("float32")
        # data = np.reshape(data, (1, im_height, im_width, 3))
        img = np.transpose(img, (2, 0, 1))
        return img / 255.0


    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)

    def next(self) -> typing.Optional[DataLabelT]:
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchCoco(dataset.Coco):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.CocoDetection(
            path.join(utils.MRT_DATASET_ROOT, "coco/val2017"),
            transform=self._to_sensor)
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def _to_tensor(self, img: Image.Image):
        img = img.resize(self._img_size)
        img = np.array(img).astype("float32")
        img = np.transpose(img, (2, 0, 1))
        return img / 255.0

    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchVoc(dataset.Voc):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.VOCDetection(
            path.join(utils.MRT_DATASET_ROOT, "voc/VOC2012/JPEGImages"),
            transform=self._to_sensor)
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def _to_tensor(self, img: Image.Image):
        img = img.resize(self._img_size)
        img = np.array(img).astype("float32")
        img = np.transpose(img, (2, 0, 1))
        return img / 255.0

    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None

class TorchCifar10(dataset.Cifar10):
    def __init__(self, batch_size = 1, img_size=(28, 28)):
        self._img_size = img_size
        val_data = tv.datasets.CIFAR10(
            path.join(utils.MRT_DATASET_ROOT, "cifar10/test_batch.bin"),
            transform=self._to_sensor)
        self.data_loader = DataLoader(
            val_data, batch_size=batch_size)
        self._max = len(self.data_loader)
        self.reset()

    def _to_tensor(self, img: Image.Image):
        img = img.resize(self._img_size)
        img = np.array(img).astype("float32")
        img = np.transpose(img, (2, 0, 1))
        return img / 255.0

    
    def __len__(self):
        return self._max

    def reset(self):
        self._iter = iter(self.data_loader)
    
    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None
