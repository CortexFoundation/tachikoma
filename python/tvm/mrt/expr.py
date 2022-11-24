from __future__ import annotations
import typing

import copy

import tvm
from tvm import relay, ir
from tvm.ir.expr import *
from tvm.relay.expr import *

from .utils import *

Parameters = typing.Dict[str, tvm.nd.NDArray]

#  def _topo_sort(expr: RelayExpr, sym_list: typing.List[RelayExpr]):
#      if sym_list.count(expr) > 0:
#          return
#      for c in expr.args:
#          _topo_sort(c, sym_list)
#      sym_list.append(symbol)

Transformer = typing.Callable[
        [RelayExpr], typing.Optional[RelayExpr]]
""" Expr Transformer

    Return new expr to transform old expr into updated one,
        or just return None for expr visit.
"""

def transform(expr: RelayExpr, callback: Transformer) -> RelayExpr:
    expr_list: typing.List[RelayExpr] = []
    def _collect_expr(expr: RelayExpr):
        # primitive ir operators, wrapper by CallNode
        if isinstance(expr, ir.op.Op):
            return
        expr_list.append(expr)
    relay.analysis.post_order_visit(expr, _collect_expr)
    #  _topo_sort(expr, expr_list)

    sym_map = {}
    for sym in expr_list:
        args = [sym_map[c] for c in sym.args]
        sym = clone(sym, args=args)
        # pre-clone symbol, to avoid misleading usage in callback
        out = callback(sym) or sym
        assert isinstance(out, Symbol)
        sym_map[sym] = out
    return sym_map[expr]

def simple_raw_print(expr: RelayExpr, params: Parameters = {}):
    info = { "op": 0, "param": 0 }
    def _simple_visit(sym):
        if not is_operator(sym):
            print("{:68} /* attrs */ \t{}".format(
                sym.name, sym.attrs))
            if is_param(sym, params):
                info["param"] += utils.product(sym.attrs["shape"])
            return

        info["op"] += 1
        print("{:15} = {:>20}{:30} /* attrs */ \t{}".format(
            sym.name, sym.op_name,
            "(" + ", ".join([i.name for i in sym.args]) + ")",
            sym.attrs,
        ))
    transform(expr, _simple_visit)
    print("="*50)
    print("Operators: {} | Parameters: {}".format(
        info["op"], info["param"]))
    print("="*50)

def transform_operators(op_names: typing.Union[typing.List[str], str]):
    if isinstance(op_names, str):
        op_names = [ op_names, ]

    def _pass(f: Transformer) -> Transformer:
        @wraps(f)
        def _wrapper(expr: RelayExpr):
            if op_name(expr) not in op_names:
                return
            return transform(expr, f)
        return _wrapper
    return _pass

def op_name(expr: RelayExpr):
    if isinstance(expr, Call):
        return expr.op.name
    elif isinstance(expr, TupleGetItem):
        return "TupleGetItem"
    elif isinstance(expr, Var):
        return "null"
    assert False

def name(expr: RelayExpr):
    if isinstance(expr, Var):
        return expr.name_hint
    return N.n()

def is_operator(expr: RelayExpr, params: Parameters = {}):
    return op_name(expr) != "null"
def is_variable(expr: RelayExpr, params: Parameters = {}):
    return op_name(expr) == "null"
def is_param(expr: RelayExpr, params: Parameters):
    return is_variable(expr) and name(expr) in params
def is_input(expr: RelayExpr, params: Parameters):
    return is_variable(expr) and name(expr) not in params

def clone(expr: RelayExpr, **kwargs) -> RelayExpr:
    expr = copy.copy(expr)
    for k, v in kwargs.items():
        setattr(expr, k, v)



