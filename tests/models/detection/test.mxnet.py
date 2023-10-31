import os
from os import path
import sys

ROOT = os.getcwd()
sys.path.insert(0, path.join(ROOT, "python"))

import tvm
from tvm import relay, ir

import numpy as np

batch_size = 16
image_shape = (3, 512, 512)
data_shape = (batch_size,) + image_shape

import torch
import torchvision as tv
data_transform = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(image_shape[1]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        [0.485,0.456,0.406], [0.229,0.224,0.225])
])
dataset = tv.datasets.ImageFolder(
        '~/.mxnet/datasets/imagenet/val',
        transform=data_transform)
test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, # set dataset batch load
        )

# use mrt wrapper to uniform api for dataset.
from tvm.mrt.dataset_torch import TorchWrapperDataset
ds = TorchWrapperDataset(test_loader)

# model inference context, like cpu, gpu, etc.
config = {
        "device": tvm.runtime.cuda(1),
        "target": tvm.target.Target("cuda -arch=sm_86")}

model_name = "mxnet_ssd_512_resnet50_v1_voc"
model_name = "faster_rcnn_resnet50_v1b_voc"
model_name = "yolo3_darknet53_voc"
model_name = "ssd_512_resnet50_v1_voc"

# with default params
from gluoncv import model_zoo
import mxnet
model = model_zoo.get_model(model_name, pretrained=True)
#input_data = np.random.randn(*data_shape)
#input_data = torch.randn(data_shape)

mod, params = relay.frontend.from_mxnet(model, {"data": data_shape})

# MRT Procedure
mod: tvm.IRModule = mod
func: relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

from tvm.mrt import stats, calibrate
from tvm.mrt.trace import Trace
from tvm.mrt.config import Pass

tr = Trace.from_expr(expr, params, model_name=model_name)
tr.bind_dataset(ds, stats.ClassificationOutput).log()

# tr.validate_accuracy(max_iter_num=20, **config)
# sys.exit()

fuse_tr = tr.fuse().log()

from tvm.mrt import segement
# Pass(log_before=True, log_after=True).register_global()
seg_tr = fuse_tr.checkpoint_run(
        segement.Spliter.get_transformer(),
        # force=True
        )
out_tr = seg_tr.checkpoint_run(
        segement.Merger.get_transformer(),
        spliter=seg_tr.symbol)
sys.exit()

with Pass(log_before=True, log_after=True):
    calib_tr = fuse_tr.calibrate(
            force=True,
            sampling_func=calibrate.SymmetricMinMaxSampling.sampling,
            ).log()

with Pass(log_before=True, log_after=True):
    dis_tr = calib_tr.quantize().log()

sim_tr = dis_tr.export().log()
sim_clip_tr = dis_tr.export(with_clip=True).log()
sim_round_tr = dis_tr.export(with_round=True).log()
sim_quant_tr = dis_tr.export(
        with_clip=True, with_round=True).log()

circom_tr = dis_tr.export(force=True, use_simulator=False).log()

tr.validate_accuracy(
        sim_tr,
        sim_clip_tr,
        sim_round_tr,
        sim_quant_tr,
        max_iter_num=20,
        **config)
sys.exit()

