from os import path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision as tv

from . import dataset, utils

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

    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            raise e
            print("error:", e)
            return None, None
