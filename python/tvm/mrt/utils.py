from __future__ import annotations

import typing
import threading
from os import path

from tvm import relay, ir

from .types import *

ROOT = path.abspath(path.join(__file__, "../../../"))
PY_ROOT = path.join(ROOT, "python")

def product(shape: ShapeT):
    total = 1
    for s in shape:
        total *= s
    return total

class N:
    def __init__(self, name=""):
        self.counter = 0
        self.scope_name = name
        self.lock = threading.Lock()
        self.last_scope = N.__GLOBAL_INSTANCE__

    def __enter__(self):
        self._set_name_scope(self)
        return self

    def __exit__(self, *args):
        self._set_name_scope(self.last_scope)

    def _alloc_name(self, prefix, suffix):
        with self.lock:
            index = self.counter
            self.counter += 1
        name = "{}{}{}".format(prefix, index, suffix)
        if self.scope_name:
            name = "{}.{}".format(self.scope_name, name)
        return name

    __GLOBAL_INSTANCE__ = None

    @staticmethod
    def _set_name_scope(ins):
        N.__GLOBAL_INSTANCE__ = ins

    @staticmethod
    def n(prefix="%", suffix=""):
        ins = N.__GLOBAL_INSTANCE__
        if ins is None:
            raise RuntimeError("Namescope not specified")
        return ins._alloc_name(prefix, suffix)

    @staticmethod
    def register_global_scope(name=""):
        N._set_name_scope(N(name))

# def get_op_name(expr: ir.RelayExpr):
#     """Get the operator name from an expression."""
#     if isinstance(expr, ir.Op):
#         return expr.name
#     if isinstance(expr, relay.expr.Call):
#         return get_op_name(expr.op)
#     if isinstance(expr, relay.TupleGetItem):
#         return get_op_name(expr.tuple_value)
#     if isinstance(expr, relay.Tuple):
#         return get_op_name(expr.fields[0])
#     if isinstance(expr, relay.expr.Var):
#         return "null"
#     return ""


# def get_args(expr):
#     """Get the arguments from an expression."""
#     if isinstance(expr, relay.expr.Call):
#         return expr.args
#     if isinstance(expr, relay.TupleGetItem):
#         return get_args(expr.tuple_value)
#     if isinstance(expr, relay.Tuple):
#         return [arg for args in map(get_args, expr.fields) for arg in args]
#     return []


# def get_attrs(expr):
#     """Get the attributes from an expression."""
#     if isinstance(expr, relay.expr.Call):
#         return expr.attrs
#     if isinstance(expr, relay.TupleGetItem):
#         return get_attrs(expr.tuple_value)
#     return {}
