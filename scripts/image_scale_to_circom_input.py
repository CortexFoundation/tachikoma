import os
from os import path
import sys
import numpy as np
from PIL import Image

ROOT = os.getcwd()
sys.path.insert(0, path.join(ROOT, "python"))

if len(sys.argv) != 3:
    print(len(sys.argv), "python", sys.argv[0], "path_to_image.png", "path_to_jsonfile.json")
    sys.exit(-1)

# target shape
batch_size = 1
image_shape = (28, 28)
input_name = "Input_I_0"
data_shape = (batch_size,) + image_shape

#img = Image.new(mode="RGB", size=(image_shape), color="green")
#input_data = np.random.randn(*data_shape)
#img = Image.fromarray(input_data)
#print("image array:", np.asarray(img))

# load rgb image, convert to gray
img = Image.open(sys.argv[1], mode="r").convert('L')
input_data = np.asarray(img)
import tvm
tvm.nd.NDArray
print("image array:", input_data)
print("image shape:", input_data.shape)
assert(image_shape == input_data.shape)

from tvm.mrt.symbol import Symbol
from tvm.mrt.transform import  WithParameters
symbols_ = Symbol(name=input_name,op_name="var",args=[], attrs={}, extra_attrs={"shape": image_shape, "dtype": "uint8"}) #"float32"})
params_ = {input_name: tvm.nd.array(input_data)} #WithParameters(name="input",op_name="var",args=[], attrs={}, extra_attrs={}, parsed=None, params={"input":tvm.nd.array(input_data)})

from tvm.mrt.trace import Trace
#from tvm.mrt.symbol import *
tr_ = Trace(name='input_trace', symbol=symbols_, params=params_)
tr_.print(short=False)

from tvm.mrt.calibrate import Calibrator, SymmetricMinMaxSampling

calib_tr = tr_.checkpoint_transform(
        Calibrator.apply(random_config={
            "enabled": True,
            "absmax": 1.0, }),
        tr_name = "input_calibrator",
        print_bf=True, print_af=True,
)

from tvm.mrt.rules import slm
from tvm.mrt.quantize import Quantizer

dt_tr: Trace = calib_tr.checkpoint_transform(
        SymmetricMinMaxSampling.apply(),
        slm.SymmetricLinearDiscretor.apply(),
        tr_name = "input_slm",
)
dt_tr = dt_tr.checkpoint_transform(
        Quantizer.apply(),
        tr_name = "input_quantizer",
        # print_bf=True, print_af=True,
)

assert(sys.argv[2].endswith(".json"))
output_name = sys.argv[2].rstrip(".json") + ".with_input.json"
print(">>> Generated, dump to {} ...".format(output_name))
with open(output_name, "w") as f_out:
    import json
    f_in = open(sys.argv[2], "r")
    input_json = json.load(f_in)
    input_json[input_name] = params_[input_name].numpy().tolist() #params_.numpy().tolist() #["1","2","3",4,5,6]
    f_out.write(json.dumps(input_json, indent=2))
    f_in.close()
