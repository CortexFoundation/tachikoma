import tvm
from tvm import relay, ir
from tvm.relay import testing
from tvm.mrt.utils import *

from tvm.mrt import runtime
from tvm.mrt import stats
from tvm.mrt import utils

import sys
import numpy as np

batch_size = 1

def load_model_from_mx() -> (ir.IRModule, ParametersT):
    import mxnet as mx
    spath, ppath = gluon.save_model("resnet18_v1", ctx=mx.cpu())
    print(spath, ppath)
    symbol, params = gluon.load_model(spath, ppath)
    return relay.frontend.from_mxnet(symbol, arg_params=params)



if False:
    num_class = 10
    image_shape = (1, 28, 28)
    mod, params = testing.mlp.get_workload(
            num_classes=num_class,
            image_shape=image_shape,
            batch_size=batch_size)
else:
    num_class = 1000
    image_shape = (3, 224, 224)
    out_shape = (batch_size, num_class)
    #  mod, params = load_model_from_mx()
    #  mod, params = testing.resnet.get_workload(
    #          batch_size=batch_size,
    #          num_classes=num_class,
    #          num_layers=18,
    #          image_shape=image_shape,)

data_shape = (batch_size,) + image_shape

def load_model_from_torch() -> (ir.IRModule, ParametersT):
    import torch
    from torchvision import models

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model = model.eval()
    input_data = torch.randn(data_shape)
    script_module = torch.jit.trace(model, [input_data]).eval()
    return relay.frontend.from_pytorch(
            script_module, [ ("input", data_shape) ])

mod, params = load_model_from_torch()

mod: tvm.IRModule = mod
func: relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

#  expr.simple_raw_print(mod["main"].body, params)


relay.Var
relay.var
relay.nn.conv2d
relay.nn.batch_flatten
relay.nn.batch_norm
relay.Tuple
relay.TupleGetItem
relay.expr.TupleWrapper
ir.tensor_type.TensorType
ir.type.TupleType

from tvm.mrt.trace import Trace, SetInputShape
from tvm.mrt.opns import *
from tvm.mrt.symbol import *
tr = Trace.from_expr(expr, params, model_name="resnet18_v1")
tr.checkpoint("init")

tr = tr.set_input_shape(
        shape=(16, *data_shape[1:]),
        checkpoint=True,)
tr.print(short=True)

# tr.print()

from tvm.mrt import fuse
from tvm.mrt import op
fuse_tr = tr.checkpoint_transform(
        fuse.FuseTupleGetItem.apply(),
        fuse.FuseBatchNorm.apply(),
        fuse.FuseAvgPool2D.apply(),
        tr_name = "fuse",
        # force=True,
        )

from tvm.mrt.calibrate import Calibrator, SymmetricMinMaxSampling

calib_tr = fuse_tr.checkpoint_transform(
        Calibrator.apply(random_config={
            "enabled": True,
            "absmax": 1.0, }),
        print_bf=True, print_af=True,
)
# calib_tr.print()
# print(type(calib_tr.symbol))

from tvm.mrt.rules import symmetric_linear_minmax as dt
from tvm.mrt.quantize import Quantizer

# calib_tr = calib_tr.subgraph(onames=["%5"])
dt_tr = calib_tr.checkpoint_transform(
        SymmetricMinMaxSampling.apply(),
        dt.SymmetricLinearDiscretor.apply(),
        # force=True,
        )
dt_tr = dt_tr.checkpoint_transform(
        Quantizer.apply(),
        # print_af=True,
        # force=True,
)
# dt_tr.print()
# dt_tr.print(short=True, suffix_layers=10)
# dt_tr.print(selects=["nn.max_pool2d"])

from tvm.mrt.fixed_point import FixPoint
# qt_tr = dt_tr.checkpoint_transform(
#         MapRequant.apply(),
#         force=True)

qt_tr = dt_tr.checkpoint_transform(
        FixPoint.apply(),
        print_bf = True, print_af = True,
        # force=True,
)

qt_tr.print(short=True, prefix_layers=20)

# qt_expr = qt_tr.to_expr()
# print(qt_expr)
# print(qt_expr.astext(show_meta_data=False))

from PIL import Image
from tvm.contrib.download import download_testdata
def get_real_image(im_height, im_width) -> np.ndarray:
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    data = np.array(image).astype("float32")
    data = np.reshape(data, (1, im_height, im_width, 3))
    data = np.transpose(data, (0, 3, 1, 2))
    data = data / 255.0
    return data

# test accuracy
from tvm.mrt import dataset
data = get_real_image(*image_shape[1:])
res = tr.run(data,
        device=tvm.runtime.cuda(1),
        target=tvm.target.cuda(),
)

out = stats.ClassificationOutput()
out.merge([res[0], np.array([0,])])
imagenet = dataset.ImageNet()
out.dl_info(imagenet.labels)

import torch
from torch.utils.data import DataLoader
import torchvision as tv

def to_tensor(img: Image.Image):
    print(img.shape)
    img = img.resize(image_shape[1:])
    img = np.array(img).astype("float32")
    # data = np.reshape(data, (1, im_height, im_width, 3))
    img = np.transpose(img, (2, 1, 0))
    return img / 255.0

val_data = tv.datasets.ImageFolder(
        path.join(utils.MRT_DATASET_ROOT, "imagenet/val"),
        transform=to_tensor)
data_loader = DataLoader(val_data, batch_size=1)
class TorchImageNet(dataset.ImageNet):
    def __init__(self):
        self.data_loader = data_loader
        self._max = len(self.data_loader)
        self.reset()

    def reset(self):
        self._iter = iter(self.data_loader)

    def next(self):
        try:
            data, label = next(self._iter)
            return data.numpy(), label.numpy()
        except Exception as e:
            print(e)
            return None, None

torch_dataset = TorchImageNet()
data, label = torch_dataset.next()
print(data.shape, label.shape)
sys.exit(-1)

tr_func = tr.populate()
runtime.multiple_validate(
        tr_func,
        dataset=TorchImageNet(),
        stats_type=stats.ClassificationOutput,
)

# tr.print()
# outs = tr.calibrate()
# print(outs.keys())

# tr_eval = tr.eval(ctx)
# runtime.multiple_validate(tr_eval, TorchImageNet(),
#         stats.ClassificationOutput,)

# fuse pass: fold_constant, fuse_batch_norm, quantize

# compare accuracy

# to_cvm

# for k, v in params.items():
#     print(k, type(v))
#     continue
# set show_meta_data=True if you want to show meta data
# print(mod.astext(show_meta_data=False))

# @ir.transform.module_pass(opt_level=2)
# def transform(mod, ctx):
#     tp = relay.TensorType((10,), "float32")
#     x = relay.var("x", tp)
#     func = relay.Function([x], relay.abs(x))
#     gv = relay.GlobalVar("myabs")
#     # new_mod = tvm.IRModule({gv: func})
#     new_mod = tvm.IRModule()
#     new_mod["myabs"] = func
#     new_mod.update(mod)
#     return new_mod

# print(relay.analysis.all_vars(mod["main"]))

# module_pass = transform
# assert isinstance(module_pass, ir.transform.ModulePass)
# assert module_pass.info.opt_level == 2

x = relay.var("x", shape=(1, 3, 28, 28), dtype="float32")
y = relay.var("y", shape=(28,), dtype="float32")
out = x + y
out = relay.abs(out)
a = relay.Constant(tvm.nd.array(np.ones((28,), dtype="float32")))
b = relay.Constant(tvm.nd.array(np.ones((28,), dtype="float32")))
c = a + b
out = out + c
relay.analysis.post_order_visit(out, _collect_ops)

mod = tvm.IRModule()
mod["main"] = relay.Function([x, y], out)
mod = relay.transform.FoldConstant()(mod)

print(mod.astext(show_meta_data=False))
sys.exit(1)

# mod = tvm.IRModule()
# mod["main"] = relay.Function([x, y], out)
# print(str(mod))

# mod = module_pass(mod)
# print("2", str(mod))

# # out = mod["myabs"](out)
# # mod["main"] = relay.Function([x, y], out)
# # print("1", str(mod))

# # mod = create_relay_module_from_model() # Output: Figure 1
import pprint
from tvm.relay.op.contrib import register
from tvm.relay.op.contrib import cvm
pattern_table = register.get_pattern_table("cvm")
pprint.pprint([p[0] for p in pattern_table])
mod = relay.transform.MergeComposite(pattern_table)(mod)
#  mod = relay.transform.AnnotateTarget(["dnnl"])(mod) # Output: Figure 2
#  mod = relay.transform.MergeCompilerRegions()(mod) # Output: Figure 3
#  mod = relay.transform.PartitionGraph()(mod) # Output: Figure 4
print("3", mod.astext(show_meta_data=False))
