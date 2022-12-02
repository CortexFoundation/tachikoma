""" op names and helper function
"""

from .symbol import *
from .utils import *

VAR_NAME = "var"
TUPLE_GET_ITEM_NAME = "TupleGetItem"
TUPLE_NAME = "Tuple"
CONV2D = "nn.conv2d"
BIAS_ADD = "nn.bias_add"
AVG_POOL2D = "nn.adaptive_avg_pool2d"
ADD = "add"
SUB = "sub"
MUL = "multiply"
BATCH_NORM = "nn.batch_norm"

def variable(X, name=None, shape=None, dtype=None):
    return X.clone(
            name = name or self.name,
            op_name = VAR_NAME,
            args = [], attrs = {
                "shape": shape or self.shape,
                "dtype": dtype or self.dtype,
                "name_hint": name or self.name,
            })

def bias_add(X: Symbol, B: Symbol, axis) -> Symbol:
    assert X.dtype == B.dtype
    return X.clone(
            name=N.n(), op_name=BIAS_ADD,
            args=[ X, B ],
            attrs={ "shape": X.shape, "dtype": X.dtype },
            )

def _broadcast_shape(ashp: ShapeT, bshp: ShapeT) -> ShapeT:
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

def _bin_op(A: Symbol, B: Symbol, op_name) -> Symbol:
    assert isinstance(other, Symbol)
    assert self.dtype == other.dtype
    oshape = _broadcast_shape(self.shape, other.shape)
    return self.clone(
            name=N.n(), op_name=op_name,
            args=[ self, other ],
            attrs={ "shape": oshape, "dtype": self.dtype },
            )


def add(A: Symbol, B: Symbol) -> Symbol:
    return _bin_op(A, B, ADD)
def sub(A: Symbol, B: Symbol) -> Symbol:
    return _bin_op(A, B, SUB)
def mul(A: Symbol, B: Symbol) -> Symbol:
    return _bin_op(A, B, MUL)

