from .circom import *
import numpy as np
# from . import model, utils
from ..symbol import *

def register_op_map(op_name):
    def wrapper_func(f):
        def _wrapper(sym: Symbol):
            if sym.op_name != op_name:
                return {}
            return {sym.op_name: f(sym)}
        return _wrapper
    return wrapper_func

def map_binary_op(sym: Symbol, name) -> str:
    A_shape = list(sym.args[0].shape)
    B_shape = list(sym.args[1].shape)
    if A_shape == B_shape:
        return "Element{}D{}".format(len(A_shape), name)

    assert len(A_shape) == len(B_shape)
    max_dim = max(len(A_shape), len(B_shape))
    #  A_shape = [1]*max_dim + A_shape
    #  B_shape = [1]*max_dim + B_shape
    equal_dims = []
    for i, (sa, sb) in enumerate(zip(A_shape[-max_dim:], B_shape[-max_dim:])):
        if sa == sb and sa != 1:
            equal_dims.append(i)
        else:
            assert any([sa == 1, sb == 1])
    assert len(equal_dims) == 1, "{}: {} vs. {}".format(
            equal_dims, A_shape, B_shape)
    return "Broadcast{}DAxis{}{}".format(
            max_dim, equal_dims[0], name)

@register_op_map("subtract")
def map_subtract(sym: Symbol):
    return map_binary_op(sym, "Sub")

@register_op_map("add")
def map_add(sym: Symbol):
    return map_binary_op(sym, "Add")

def map_component(sym: Symbol) -> CircomGenerator:
    inputs = sym.args
    comp_map = {
        # "null": "Input",
        "var": "Input",

        "nn.pad_scalar": "Pad2D",
        "nn.conv2d": "Conv2D_CHW",
        "nn.dense": "Dense2" if len(inputs) == 2 else "Dense",

        **map_subtract(sym),
        **map_add(sym),
        #  "add": "ElementAdd",

        "mul_scalar": "MulScalar",
        "add_scalar": "AddScalar",
        "subtract_scalar": "SubScalar",

        "right_shift": "RightShift",
        "clip": "Clip",

        "image.resize2d": "Resize2D",

        "reshape": "ReShape{}D".format(len(sym.shape)),
        #  "reshape": "ReShape" + str(len(sym.attrs["shape"])) + "D",
        "flatten": "Flatten{}D".format(
            len(inputs[0].shape) if inputs else 0),
    }
    return components[comp_map[sym.op_name]]

#  def shape_adapter(generator: CircomGenerator):
#      """ Change model to adapt circom shape rules by reshape & flatten. """
#      def prod(shape):
#          total = 1
#          for s in shape:
#              total *= s
#          return total

#      count = 0
#      inputs = []
#      for idx, inp in enumerate(generator.inputs):
#          expected_dim = generator.comp.input_dims[idx]
#          if expected_dim == len(inp.shape):
#              pass
#          elif expected_dim == 1:
#              flatten_shape = prod(inp.shape)
#              comp = components["Flatten" + str(len(inp.shape)) + "D"]
#              inp = comp("flatten_" + str(count),
#                      [ inp, ], { "shape": ( flatten_shape, )})
#              count += 1
#              generator
#          else:
#              print("cannot automatically adapt shape")
#              raise RuntimeError(generator.info())

#          inputs.append(inp)


def model2circom(symbol, params):
    generator_map: typing.Dict[str, CircomGenerator] = {}
    circom_ops = set()
    def sym2circom(sym: Symbol):
        name = sym.name
        if name in generator_map:
            return

        inputs = [generator_map[i.name] for i in sym.args]

        attrs = {k: v for k, v in sym.attrs.items()}
        attrs.update(sym.extra_attrs)
        gen = map_component(sym)(name, inputs, attrs)
        circom_ops.add(gen.comp.op_name)
        #  comp.fill_circom()
        generator_map[name] = gen

    visit(symbol, sym2circom)
    #  print("Invoked Circoms: \n", "\n".join(circom_ops))

    out = components["Output"](
            "out", [generator_map[symbol.name]],
            { "shape": generator_map[symbol.name].shape })
    return out

def input_json(
        symbol: Symbol,
        params: typing.Dict[str, np.ndarray]):
    """ ndarray of str in json format, instead of int """
    def _as_str_data(data):
        if isinstance(data, list):
            return [_as_str_data(d) for d in data]
        assert isinstance(data, int)
        return str(data)

    return {k: _as_str_data(v.tolist()) \
            for k, v in params.items()}


def _shape_adapter(sym: Symbol):
    supported_ops = [
        "clip",
        "mul_scalar", "add_scalar", "subtract_scalar",
        "right_shift",
    ]
    if sym.op not in supported_ops:
        inputs = []
        for inp in sym.inputs:
            if "orig_shape" in inp.attrs:
                inp = Symbol("reshape", inp.name + "_r",
                        [ inp, ], [],
                        { "shape": inp.attrs["orig_shape"],
                            "dtype": sym.attrs["dtype"],
                        })
            inputs.append(inp)
        sym = sym.clone(inputs=inputs)
        return sym

    if len(sym.attrs["shape"]) == 1:
        return

    input_shape = list(sym.inputs[0].attrs["shape"])
    orig_shape = list(sym.inputs[0].attrs.get(
        "orig_shape", input_shape))
    shape_one = utils.product(input_shape)

    inputs = []
    for inp in sym.inputs:
        shape = list(inp.attrs["shape"])
        assert input_shape == shape
        if len(shape) != 1:
            inp = Symbol("flatten", sym.inputs[0].name + "_f",
                    [ sym.inputs[0], ], [],
                    { "shape": ( shape_one, ),
                        "dtype": sym.attrs["dtype"], })
        inputs.append(inp)

    #  assert list(sym.attrs["shape"]) == input_shape, sym.info()
    sym = sym.clone(inputs=inputs)
    sym.attrs.update({
        "shape": ( shape_one, ),
        "orig_shape": orig_shape,
        })
    return sym

def shape_adapter(symbol: Symbol):
    symbol = visit(symbol, _shape_adapter)

    if "orig_shape" in symbol.attrs:
        symbol = Symbol("reshape", symbol.name + "_r",
                [ symbol, ], [],
                {
                    "shape": symbol.attrs["orig_shape"],
                    "dtype": symbol.attrs["dtype"],
                })

    def _clean_attrs(sym: Symbol):
        if "orig_shape" in sym.attrs:
            del sym.attrs["orig_shape"]
    symbol = visit(symbol, _clean_attrs)
    return symbol
