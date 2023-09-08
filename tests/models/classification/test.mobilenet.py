import os
import sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np
from PIL import Image
import torch

import tvm
from tvm import ir
from tvm.mrt.utils import *
from tvm.mrt import runtime
from tvm.mrt import stats, dataset
from tvm.mrt import utils

batch_size = 16
image_shape = (3, 28, 28)
data_shape = (batch_size,) + image_shape

def load_model_from_torch() -> (ir.IRModule, ParametersT):
    from torchvision import models

    model = models.mobilenet_v2(weights='DEFAULT')
    model = model.eval()
    input_data = torch.randn(data_shape)
    script_module = torch.jit.trace(model, [input_data]).eval()
    return tvm.relay.frontend.from_pytorch(
            script_module, [ ("input", data_shape) ])

mod, params = load_model_from_torch()
mod: tvm.IRModule = mod
func: tvm.relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

from tvm.mrt.trace import Trace
from tvm.mrt.opns import *
from tvm.mrt.symbol import *
tr = Trace.from_expr(expr, params, model_name="mobilenet_v2")
#  tr = tr.subgraph(onames=["%11"])
tr.checkpoint()
tr.log()
#  tr.print(short=True, param_config={ "use_all": True, })

from tvm.mrt import fuse
from tvm.mrt import op
from tvm.mrt.calibrate import Calibrator, SymmetricMinMaxSampling

#  const_tr = tr.checkpoint_transform(
#          force=True,)
#  const_tr.log()

fuse_tr = tr.checkpoint_transform(
        fuse.FuseConstant.apply(),
        #  fuse.FuseTupleGetItem.apply(),
        #  fuse.FuseDropout.apply(),
        #  fuse.FuseBatchNorm.apply(),
        tr_name = "fuse",
        force=True,
        )
fuse_tr.log()
#  fuse_tr.print(short=True)

fuse_tr = fuse_tr.checkpoint_transform(
        fuse.FuseAvgPool2D.apply(),
        fuse.FuseNaiveSoftmax.apply(),
        tr_name = "fuse-post",
        #  force=True,
        )
fuse_tr.log()

from tvm.mrt.dataset_torch import TorchImageNet
ds = TorchImageNet(
        batch_size=batch_size,
        img_size=image_shape[1:],)
data, _ = ds.next()

#  calib_tr = fuse_tr.checkpoint_transform(
#          Calibrator.apply(data=tvm.nd.array(data)),
#          print_af=True,
#          force=True,
#  )

fuse_tr = fuse_tr.checkpoint_transform(
        fuse.FuseTupleGetItem.apply(),
        fuse.FuseDropout.apply(),
        fuse.FuseBatchNorm.apply(),
        fuse.FuseNaiveMathmatic.apply(),
        )
fuse_tr.log()


calib_tr = fuse_tr.checkpoint_transform(
        Calibrator.apply(data=tvm.nd.array(data)),
        #  Calibrator.apply(random_config={
        #      "enabled": True,
        #      "absmax": 1.0, }),
        #  print_bf=True,
        print_af=True,
        force=True,
)

sample_tr = calib_tr.checkpoint_transform(
        SymmetricMinMaxSampling.apply(),
        #  print_bf=True,
        print_af=True,
        #  force=True,
        )
sample_tr.log()

#  from tvm.mrt.precision import PrecisionAnnotator

#  prec_tr = sample_tr.checkpoint_transform(
#          PrecisionAnnotator.apply(),
#          print_bf=True, print_af=True,
#          )
#  prec_tr.log()

from tvm.mrt.discrete import Discretor2
dis_tr = sample_tr.checkpoint_transform(
        Discretor2.apply(),
        print_bf=True, print_af=True,
        force=True,
        )
dis_tr.log()
sys.exit(0)


from tvm.mrt.rules import slm
from tvm.mrt.quantize import Quantizer

dt_tr = calib_tr.checkpoint_transform(
        slm.SymmetricLinearDiscretor.apply(),
        )
# dt_tr.print(short=True)
dt_tr: Trace = dt_tr.checkpoint_transform(
        Quantizer.apply(),
        # print_bf=True, print_af=True,
        # force=True,
)

from tvm.mrt.fixed_point import FixPoint, Simulator
sim_tr = dt_tr.checkpoint_transform(
        Simulator.apply(),
        # force=True,
        )
# sim_tr.log()
# sim_tr.print(short=True)

qt_tr = dt_tr.checkpoint_transform(
        FixPoint.apply(),
        # print_bf = True, print_af = True,
        # force=True,
)
qt_tr.print(short=False)

from tvm.mrt.zkml import circom, transformer, model as ZkmlModel

symbol, params = qt_tr.symbol, qt_tr.params
print(">>> Start circom gen...")
symbol, params = ZkmlModel.resize_batch(symbol, params)
#ZkmlModel.simple_raw_print(symbol, params)
symbol, params = transformer.change_name(symbol, params)
# set input as params
symbol_first = ZkmlModel.visit_first(symbol)
#print(">>> before circom gen ...", symbol_first, symbol_first.is_input(), symbol_first.is_param())
input_data = torch.randint(255, image_shape)
params[symbol_first.name] = input_data
circom_out, circom_gen_map = transformer.model2circom(symbol, params)
print(">>> Generating circom code ...")
circom_code = circom.generate(circom_out)
print(">>> Generating circom input ...")
input_json = circom.input_json(circom_gen_map, params)

output_name = "circom_model_test"
print(">>> Generated, dump to {} ...".format(output_name))
with open(output_name + ".circom", "w") as f:
    f.write(circom_code)
with open(output_name + ".json", "w") as f:
    import json
    f.write(json.dumps(input_json, indent=2))

print(">>> success exit sys +1 <<<")
sys.exit(+1)
