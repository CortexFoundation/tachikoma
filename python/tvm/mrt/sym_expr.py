"""
==============================================================
API from relay.Function to Symbol.
==============================================================
"""

from tvm import relay, ir
from tvm.ir.expr import *
from tvm.relay.expr import *

from .symbol import *
from .op import *

__ALL__ = [ "expr2symbol", "symbol2expr", ]

SUPPORTED_EXPR_TYPE = (
        relay.expr.Var,
        ir.op.Op, # Op are wrapped by Call.
        relay.expr.Call,
        relay.expr.TupleGetItem,
        )

def _expr_type(checked_type: ir.type.Type, key):
    if isinstance(checked_type, ir.type.TupleType):
        return [_expr_type(f, key) for f in checked_type.fields]
    return getattr(checked_type, key)

def expr2symbol(expr: RelayExpr) -> Symbol:
    mod = relay.transform.InferType()(ir.IRModule.from_expr(expr))
    expr = mod["main"].body

    symbol_map = {}
    def _cast_expr(node: RelayExpr):
        if not isinstance(node, SUPPORTED_EXPR_TYPE):
            raise RuntimeError(
                "MRT not support expr type:{}".format(type(node)))

        if isinstance(node, ir.op.Op):
            return

        if isinstance(node, relay.Var):
            name = node.name_hint or N.n(prefix="input_")
            symbol_map[node] = Symbol(name, VAR_NAME, [], {})
        elif isinstance(node, relay.Call):
            args = [symbol_map[i] for i in node.args]
            attrs = node.attrs or {}
            attrs = {k: attrs[k] for k in attrs.keys()}
            symbol_map[node] = Symbol(N.n(), node.op.name,
                    args, attrs)
        elif isinstance(node, relay.TupleGetItem):
            args = [ symbol_map[node.tuple_value], ]
            symbol_map[node] = Symbol(N.n(), TUPLE_GET_ITEM_NAME,
                    args, { "index": node.index })
        elif isinstance(node, relay.Tuple):
            args = [ symbol_map[f] for f in node.fields ]
            symbol_map[node] = Symbol(N.n(), TUPLE_NAME,
                    args, {})

        dtype = _expr_type(node.checked_type, "dtype")
        shape = _expr_type(node.checked_type, "concrete_shape")
        #  print(dtype, shape, type(shape))
        symbol_map[node].attrs.update({
            "shape": list(shape),
            "dtype": dtype,
        })

    with N():
        relay.analysis.post_order_visit(expr, _cast_expr)
    return symbol_map[expr]

def symbol2expr(symbol: Symbol, expr_map={}) -> RelayExpr:
    # operator creator don't need shape or dtype attrs,
    #   except for the variable.
    def _remove_type(sym: Symbol):
        if is_variable(sym):
            return

        if "shape" in sym.attrs:
            del sym.attrs["shape"]
        if "dtype" in sym.attrs:
            del sym.attrs["dtype"]
        return sym
    symbol = transform(symbol, _remove_type)

    expr_map.clear()
    def _cast_symbol(sym: Symbol):
        args = [expr_map[i] for i in sym.args]
        if sym.is_op(TUPLE_NAME):
            out = relay.Tuple(args)
        else:
            try:
                out = eval("relay." + sym.op_name)(*args, **sym.attrs)
            except Exception as e:
                print(sym, [type(a) for a in args])
                raise e

        if isinstance(out, relay.TupleWrapper):
            out = out.tuple_value
        # relay.transform.InferTypeLocal(out)
        expr_map[sym] = out

    _ = transform(symbol, _cast_symbol)
    return expr_map[symbol]
