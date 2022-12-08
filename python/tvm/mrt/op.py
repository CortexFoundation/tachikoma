""" op names and helper function
"""

from dataclasses import dataclass

from .symbol import *
from .utils import *
from .opns import *

def retrieve_operator(symbol: Symbol) -> Symbol:
    args = [ variable(a.name, a.shape, a.dtype) \
            for a in symbol.args ]
    return Symbol.base(symbol, args=args)

def infer_type(symbol: Symbol) -> Symbol:
    from tvm import relay, ir
    from tvm.mrt import sym_expr

    expr = sym_expr.symbol2expr(symbol)
    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    expr = mod["main"].body
    return sym_expr.expr2symbol(expr)

@dataclass(repr=False)
class InferType(Symbol):
    def __post_init__(self):
        assert is_operator(self)

        if type(self) is InferType:
            sym = retrieve_operator(self)
            sym = infer_type(sym)
            self.attrs.update({
                "shape": sym.shape,
                "dtype": sym.dtype,
            })
        else:
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

@dataclass(repr=False)
class FirstLikeInferType(InferType):
    def _infer_shape(self) -> ShapeT:
        return self.args[0].shape

def _new_op(op_name, *args, **attrs) -> Symbol:
    return Symbol(N.n(), op_name, args, attrs)

def _register_op(op_name,
        infer_type: typing.Type[InferType] = InferType):
    def _op(*args, **attrs) -> Symbol:
        op = _new_op(op_name, *args, **attrs)
        return infer_type.base(op)
    return _op

tuple = _register_op(TUPLE)
bias_add = _register_op(BIAS_ADD)
add = _register_op(ADD)
sub = _register_op(SUB)
mul = _register_op(MUL)
sum = _register_op(SUM)

requant = _register_op(REQUANT, FirstLikeInferType)
pclip = _register_op(PCLIP, FirstLikeInferType)
rs_pclip = _register_op(RS_PCLIP, FirstLikeInferType)

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


