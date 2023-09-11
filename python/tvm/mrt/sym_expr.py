"""
==============================================================
API from relay.Function to Symbol.
==============================================================
"""
import typing
from dataclasses import dataclass

from tvm import relay, ir, tir
from tvm.ir.expr import *
from tvm.relay.expr import *

from .symbol import *
from .opns import *
from . import op

__ALL__ = [ "expr2symbol", "symbol2expr", ]

def _expr_type(checked_type: ir.type.Type, key):
    if isinstance(checked_type, ir.type.TupleType):
        return [_expr_type(f, key) for f in checked_type.fields]
    return getattr(checked_type, key)

def _convert_to_py(value):
    if isinstance(value, ir.container.Array):
        return [ _convert_to_py(v) for v in value ]
    elif isinstance(value, ir.container.Map):
        return {k: _convert_to_py(v) for k, v in value.items()}
    elif isinstance(value, tir.expr.IntImm):
        return int(value)
    return value

def _format_containers(attrs):
    for k, v in attrs.items():
        attrs[k] = _convert_to_py(v)

def expr2symbol(expr: RelayExpr) -> Symbol:
    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    expr = mod["main"].body

    symbol_map = {}
    def _cast_expr(node: RelayExpr):
        if isinstance(node, ir.op.Op):
            return

        name, op_name, args = None, None, []
        dtype = _expr_type(node.checked_type, "dtype")
        shape = _expr_type(node.checked_type, "concrete_shape")
        extra_attrs = { "shape": shape, "dtype": dtype, }

        attrs = { "extra_attrs": extra_attrs, }
        if isinstance(node, relay.Var):
            name = node.name_hint or N.n(prefix="input_")
            symbol_map[node] = op.variable(name, shape, dtype)
        elif isinstance(node, relay.Call):
            if node.op.name == CONCAT:
                args = [ symbol_map[f] for f in node.args[0].fields ]
            else:
                args = [symbol_map[i] for i in node.args]
            nattrs = node.attrs or {}
            attrs.update({k: nattrs[k] for k in nattrs.keys()})
            _format_containers(attrs)
            symbol_map[node] = op._new_op(
                    node.op.name, *args, **attrs)
        elif isinstance(node, relay.TupleGetItem):
            args = [ symbol_map[node.tuple_value], ]
            attrs['index'] = node.index
            symbol_map[node] = op._new_op(
                    TUPLE_GET_ITEM, *args, **attrs)
        elif isinstance(node, relay.Tuple):
            args = [ symbol_map[f] for f in node.fields ]
            symbol_map[node] = op._new_op(
                    TUPLE, *args, **attrs)
        else:
            raise RuntimeError(
                "MRT not support expr type:{}".format(type(node)))


    with N():
        relay.analysis.post_order_visit(expr, _cast_expr)
    return symbol_map[expr]

def symbol2expr(symbol: Symbol, expr_map={}) -> RelayExpr:
    expr_map.clear()
    def _cast_symbol(sym: Symbol):
        args = [expr_map[i.name] for i in sym.args]

        attrs = {k: v for k, v in sym.attrs.items()}
        # operator creator don't need shape or dtype attrs,
        #   except for the variable.
        if op.is_variable(sym):
            attrs.update({
                "shape": sym.shape, "dtype": sym.dtype,
                "name_hint": sym.name,
            })

        if sym.is_op(TUPLE):
            out = relay.Tuple(args)
        elif sym.is_op(CONCAT):
            out = relay.concatenate(args, **attrs)
        else:
            try:
                out = eval("relay." + sym.op_name)(*args, **attrs)
            except Exception as e:
                print(sym, [type(a) for a in args], attrs)
                raise e

        if isinstance(out, relay.TupleWrapper):
            out = out.tuple_value
        expr_map[sym.name] = out

    visit(symbol, _cast_symbol)

    return expr_map[symbol.name]
