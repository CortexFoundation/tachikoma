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
    tv.transforms.Resize(image_shape[-1]),
    tv.transforms.CenterCrop(image_shape[1]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(
        [0.485,0.456,0.406], [0.229,0.224,0.225])
])

#  from gluoncv import data
#  from gluoncv.data.batchify import Tuple, Stack, Pad
#  from gluoncv.data.transforms import presets
#  data_transform = presets.ssd.SSDDefaultValTransform(*image_shape[1:])

#  dataset = data.ImageNet(train=False)
#  batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
#  test_loader = data.dataloader.DataLoader(
#          dataset.transform(data_transform),
#          batch_size=batch_size,
#          shuffle=False,
#          batchify_fn=batchify_fn,
#          )
# dataset = tv.datasets.VOCDetection(
#         "~/.mxnet/datasets/voc/",
#         transform=data_transform,
#         )
dataset = tv.datasets.ImageFolder(
        '~/.mxnet/datasets/imagenet/val',
        transform=data_transform)
test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, # set dataset batch load
        )

#  for i, b in enumerate(test_loader):
#      if i > 3:
#          break
#      print(b[0].shape, b[1].shape)
#  sys.exit()

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
# model_name = "ssd_512_resnet50_v1_voc"

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

#  with open("/tmp/expr.txt", "w") as f:
#      f.write(expr.astext(show_meta_data=False))

from tvm.mrt import stats, calibrate, opns
from tvm.mrt.trace import Trace
from tvm.mrt.config import Pass

tr = Trace.from_expr(expr, params, model_name=model_name)
tr.bind_dataset(ds, stats.ClassificationOutput).log()


from tvm.mrt import stats
from torchmetrics.detection import MeanAveragePrecision
class TorchStatistics(stats.Statistics):
    def __init__(self):
        self.map = MeanAveragePrecision()

    def reset(self):
        self.map = MeanAveragePrecision()

    def merge(self, dl):
        (pred_label, score, bbox), label = dl
        preds = [ dict(
            boxes=torch.from_numpy(bbox.numpy()),
            scores=torch.from_numpy(score.numpy()),
            labels=torch.from_numpy(pred_label.numpy()),
            ) ]
        target = [ dict(
            boxes=torch.from_numpy(label[0].numpy()),
            labels=torch.from_numpy(label[1].numpy()),
            ) ]
        self.map.update()


#  tr.validate_accuracy(max_iter_num=20, **config)
#  sys.exit()

#  c = Pass(log_before=True, log_after=True).register_global()
# Pass(log_before=True, log_after=True).register_global()
fuse_tr = tr.fuse(force=True).log()

# from tvm.mrt import fuse
# fuse_tr = fuse_tr.checkpoint_run(
#         fuse.FuseLeakyReLU.get_transformer(),
#         fuse.FuseDivide.get_transformer(),
#         # force=True,
#         )
# fuse_tr.log()

# import numpy as np
# from tvm.mrt import inference, op
# p = op.variable("input", (1, ), "float32")
# o = inference.run(op.exp(p), [ np.full((1, ), 2.226) ])
# print(o)
# sys.exit()

from tvm.mrt import segement
seg_tr = fuse_tr.checkpoint_run(
        segement.Spliter.get_transformer(),
        force=True
        ).log()

calib_tr = seg_tr.calibrate(
        sampling_func=calibrate.SymmetricMinMaxSampling.sampling,
        # force=True,
        ).log()
dis_tr = calib_tr.quantize(
        # force=True
        ).log()

# from tvm.mrt.precision import PrecisionRevisor
# dis_tr = dis_tr.checkpoint_run(
#         PrecisionRevisor.get_transformer(),
#         # force=True
#         ).log()
# # dis_tr = dis_tr.checkpoint_run(infer_precision)

dis_tr = dis_tr.checkpoint_run(
        segement.Merger.get_transformer(),
        spliter=seg_tr.symbol,
        force=True,
        ).log()

sim_tr = dis_tr.export().log()
sim_clip_tr = dis_tr.export(with_clip=True).log()
sim_round_tr = dis_tr.export(with_round=True).log()
sim_quant_tr = dis_tr.export(
        with_clip=True, with_round=True).log()

circom_tr = dis_tr.export(use_simulator=False).log()

tr.validate_accuracy(
        sim_tr,
        sim_clip_tr,
        sim_round_tr,
        sim_quant_tr,
        max_iter_num=20,
        **config)
sys.exit()

