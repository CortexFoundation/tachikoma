from __future__ import annotations
import typing

import copy
from functools import wraps
import pprint

import tvm
from tvm import relay, ir
from tvm.ir.expr import *
from tvm.relay.expr import *

from .types import *

def expr_type(checked_type: ir.type.Type, key):
    if isinstance(checked_type, ir.type.TupleType):
        return [expr_type(f, key) for f in checked_type.fields]
    return getattr(checked_type, key)

def update_expr_args(old: RelayExpr, expr_map) -> RelayExpr:
    op_str = op_name(old)
    args = []
    attrs = {}
    if isinstance(old, TupleGetItem):
        attrs["tuple_value"] = expr_map[old.tuple_value]
        attrs["index"] = old.index
    elif isinstance(old, Tuple):
        attrs["fields"] = [expr_map[f] for f in old.fields]
    elif isinstance(old, Call):
        args = [expr_map[a] for a in old.args]
        attrs = old.attrs or {}
        attrs = {k: attrs[k] for k in attrs.keys()}
    elif isinstance(old, Var):
        attrs["name_hint"] = old.name_hint
        attrs["dtype"] = expr_type(old.checked_type, "dtype")
        attrs["shape"] = expr_type(old.checked_type, "concrete_shape")

    try:
        new = eval("relay." + op_str)(*args, **attrs)
    except Exception as e:
        print(op_name(old))
        raise e

    if isinstance(new, relay.TupleWrapper):
        new = new.tuple_value
    return new


def clone(expr: RelayExpr, **kwargs) -> RelayExpr:
    expr = copy.copy(expr)
    for k, v in kwargs.items():
        setattr(expr, k, v)


Visitor = typing.Callable[ [RelayExpr], None ]
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

    expr_map = {}
    for i, sym in enumerate(expr_list):
        out = update_expr_args(sym, expr_map)
        # pre-clone symbol, to avoid misleading usage in callback
        out = callback(out) or out
        assert isinstance(out, RelayExpr)
        expr_map[sym] = out
    return expr_map[expr]

def infer_type(expr: RelayExpr) -> expr:
    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    return mod["main"].body

def visit(expr: RelayExpr, callback: Visitor):
    expr_list: typing.List[RelayExpr] = []
    def _collect_expr(expr: RelayExpr):
        # primitive ir operators, wrapper by CallNode
        if isinstance(expr, ir.op.Op):
            return
        expr_list.append(expr)
    relay.analysis.post_order_visit(expr, _collect_expr)

    for sym in expr_list:
        callback(sym)


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

def filter_operators(*op_names: typing.List[str]):
    def _pass(f: Transformer) -> Transformer:
        @wraps(f)
        def _wrapper(expr: RelayExpr):
            if op_name(expr) not in op_names:
                return
            return f(expr)
        return _wrapper
    return _pass

VAR_NAME = "var"
TUPLE_NAME = "Tuple"
TUPLE_GET_ITEM_NAME = "TupleGetItem"

def op_name(expr: RelayExpr):
    if isinstance(expr, Call):
        return expr.op.name
    elif isinstance(expr, TupleGetItem):
        return TUPLE_GET_ITEM_NAME
    elif isinstance(expr, Tuple):
        return TUPLE_NAME
    elif isinstance(expr, Var):
        return VAR_NAME
    assert False, type(expr)

def is_operator(expr: RelayExpr, params: Parameters = {}):
    return not isinstance(expr, Var)
def is_variable(expr: RelayExpr, params: Parameters = {}):
    return isinstance(expr, Var)
def is_param(expr: RelayExpr, params: Parameters):
    return is_variable(expr) and expr.name_hint in params
def is_input(expr: RelayExpr, params: Parameters):
    return is_variable(expr) and expr.name_hint not in params



