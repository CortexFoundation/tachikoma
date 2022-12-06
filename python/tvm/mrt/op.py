""" op names and helper function
"""

from dataclasses import dataclass

from .symbol import *
from .utils import *

VAR = "var"

TUPLE = "Tuple"
TUPLE_GET_ITEM = "TupleGetItem"

CONV2D = "nn.conv2d"
DENSE = "nn.dense"
BATCH_NORM = "nn.batch_norm"
BIAS_ADD = "nn.bias_add"
RELU = "nn.relu"
AVG_POOL2D = "nn.adaptive_avg_pool2d"
MAX_POOL2D = "nn.max_pool2d"

SQUEEZE = "squeeze"
RESHAPE = "reshape"

ADD = "add"
SUB = "sub"
MUL = "multiply"

# ======= mrt requant op ==========
REQUANT = "mrt.requant"


@dataclass
class InferType(Symbol):
    def __post_init__(self):
        assert is_operator(self)
        self.attrs.update({
            "shape": self._infer_shape(),
            "dtype": self._infer_type(),
        })

    def _infer_type(self):
        assert all([self.args[0].dtype == a.dtype \
                for a in self.args])
        return self.args[0].dtype

    def _infer_shape(self) -> ShapeT:
        raise NotImplementedError("")

class FirstLikeInferType(InferType):
    def _infer_shape(self):
        return self.args[0].shape

class BroadcastInferType(InferType):
    def _infer_shape(self):
        assert len(self.args) == 2
        A, B = self.args
        ashp, bshp = A.shape, B.shape
        alen, blen = len(ashp), len(bshp)
        olen = max(alen, blen)
        ashp = [1] * olen + list(ashp)
        bshp = [1] * olen + list(bshp)

        oshp = [1] * olen
        for i in range(olen):
            adim = ashp[i-olen]
            bdim = bshp[i-olen]
            if adim == 1 or bdim == 1:
                oshp[i] = max(adim, bdim)
            else:
                assert adim == bdim
                oshp[i] = adim
        return oshp

def _new_op(op_name, *args, **attrs) -> Symbol:
    return Symbol(N.n(), op_name, args, attrs)

def _register_op(op_name, infer_type: typing.Type[InferType]):
    def _op(*args, **attrs) -> Symbol:
        op = _new_op(op_name, *args, **attrs)
        return infer_type.base(op)
    return _op

bias_add = _register_op(BIAS_ADD, FirstLikeInferType)
add = _register_op(ADD, BroadcastInferType)
sub = _register_op(SUB, BroadcastInferType)
mul = _register_op(MUL, BroadcastInferType)

requant = _register_op(REQUANT, FirstLikeInferType)

def variable(name, shape, dtype) -> Symbol:
    """ Create varible for symbol. """
    return Symbol(name, VAR, [], {
        "shape": shape, "dtype": dtype,
        "name_hint": name,
        })

def is_operator(symbol: Symbol, params: ParametersT = {}):
    return symbol.op_name != VAR
def is_variable(symbol: Symbol, params: ParametersT = {}):
    return symbol.op_name == VAR
def is_input(symbol: Symbol, params: ParametersT):
    return is_variable(symbol) and symbol.name not in params
def is_param(symbol: Symbol, params: ParametersT):
    return is_variable(symbol) and symbol.name in params

relay.sum
relay.add


