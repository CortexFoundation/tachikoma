from __future__ import annotations
import typing

import tvm
from tvm import relay, ir
from tvm.ir.expr import *
from tvm.relay.expr import *

from dataclasses import dataclass, is_dataclass, fields
from functools import wraps

from .utils import *

VAR_NAME = "Var"
TUPLE_GET_ITEM_NAME = "TupleGetItem"
TUPLE_NAME = "Tuple"

def is_operator(symbol: Symbol, params = {}):
    return symbol.op_name != VAR_NAME

def is_variable(symbol: Symbol, params = {}):
    return symbol.op_name == VAR_NAME

def is_input(symbol: Symbol, params):
    return is_variable(symbol) and symbol.name not in params

def is_param(symbol: Symbol, params):
    return is_variable(symbol) and symbol.name in params

CopyAttrsT = typing.Union[typing.List[str], str]

@dataclass
class Symbol:
    """ Uniform Symbol Representation for RelayExpr

    RelayExpr has different format for operators, functions,
        which is hard to apply uniform transformation pass.
        Such as the `TupleGetItem`.
    """

    name: str
    op_name: str
    args: typing.List[Symbol]
    attrs: typing.Dict[str, typing.Any]

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def variable(name):
        return Symbol(name, VAR_NAME, [], { "name_hint": name })

    def as_parameter(self):
        return self.clone(
                copy_attrs = ["shape", "dtype"],
                op_name = VAR_NAME,
                args = [])

    def is_op(self, op_name):
        return self.op_name == op_name

    def clone(self, copy_attrs: CopyAttrsT=[], **kw) -> Symbol:
        kw.setdefault("attrs", {})
        if isinstance(copy_attrs, str):
            copy_attrs = [ copy_attrs, ]
        # copy all attributes by default
        copy_attrs = copy_attrs or self.attrs.keys()
        kw["attrs"].update({k: self.attrs[k] for k in copy_attrs})

        data = dict((f.name, getattr(self, f.name)) \
                for f in fields(self))
        data.update(kw)

        return Symbol(**data)

    def __eq__(self, other: Symbol):
        return self.args == other.args and hash(self) == hash(other)

    def __str__(self):
        args_info= ["{}@{}".format(
            i.name, i.attrs.get("shape", None)) \
            for i in self.args ]
        return "{} = {}({}) /* attrs */ \t{}".format(
            self.name, self.op_name,
            ", ".join(args_info),
            self.attrs)


def _topo_sort(symbol: Symbol, sym_list: typing.List[Symbol]):
    if sym_list.count(symbol) > 0:
        return
    for c in symbol.args:
        _topo_sort(c, sym_list)
    sym_list.append(symbol)

Transformer = typing.Callable[[Symbol], typing.Optional[Symbol]]
""" Symbol Transformer

    Return new symbol to transform old symbol into updated one,
        or just return None for symbol visit.
"""

def transform(symbol: Symbol, callback: Transformer) -> Symbol:
    """ Transform symbol from old to new, with inputs updated. """
    sym_list: typing.List[Symbol] = []
    _topo_sort(symbol, sym_list)

    sym_map = {}
    for sym in sym_list:
        args = [sym_map[c.name] for c in sym.args]
        sym = sym.clone(args=args)
        # pre-clone symbol, to avoid misleading usage in callback
        out = callback(sym.clone()) or sym
        assert isinstance(out, Symbol)
        sym_map[sym.name] = out
    return sym_map[symbol.name]

def transform_operators(op_names: typing.Union[typing.List[str], str]):
    if isinstance(op_names, str):
        op_names = [ op_names, ]

    def _pass(f: Transformer):
        @wraps(f)
        def _wrapper(symbol: Symbol) -> typing.Optional[Symbol]:
            if symbol.op_name not in op_names:
                return
            return transform(symbol, f)
        return _wrapper
    return _pass

def simple_raw_print(symbol: Symbol, params={}):
    info = { "op": 0, "param": 0 }
    def _simple_visit(sym):
        if is_param(sym, params):
            info["param"] += utils.product(sym.attrs["shape"])

        info["op"] += is_operator(sym)
        print("{:15} = {:>20}{:30} /* attrs */ \t{}".format(
            sym.name, sym.op_name,
            "(" + ", ".join([i.name for i in sym.args]) + ")",
            sym.attrs,
        ))
    transform(symbol, _simple_visit)
    print("="*50)
    print("Operators: {} | Parameters: {}".format(
        info["op"], info["param"]))
    print("="*50)


# ==============================================================
# API from relay.Function to Symbol.
# ==============================================================

SUPPORTED_EXPR_TYPE = (
        relay.expr.Var,
        ir.op.Op, # Op are wrapped by Call.
        relay.expr.Call,
        relay.expr.TupleGetItem,
        )

def expr2symbol(expr: RelayExpr) -> Symbol:
    symbol_map = {}
    def _cast_expr(node):
        if not isinstance(node, SUPPORTED_EXPR_TYPE):
            raise RuntimeError(
                "MRT not support expr type:{}".format(type(node)))

        if isinstance(node, relay.Var):
            name = node.name_hint or N.n(prefix="input_")
            symbol_map[node] = Symbol.variable(name)
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

    with N():
        relay.analysis.post_order_visit(expr, _cast_expr)
    return symbol_map[expr]

def symbol2expr(symbol: Symbol) -> RelayExpr:
    expr_map = {}
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
        expr_map[sym] = out

    with N():
        transform(symbol, _cast_symbol)
    return expr_map[symbol]




