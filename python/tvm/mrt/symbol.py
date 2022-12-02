from __future__ import annotations
import typing

import json
from functools import wraps
from dataclasses import dataclass, fields, is_dataclass

from .utils import *
from .types import *

__ALL__ = [
        "Symbol",
        "visit", "transform",
        "filter_operators",
        ]

_CopyAttrsT = typing.Union[typing.List[str], str]

@dataclass
class Symbol:
    """ Uniform Symbol Representation for RelayExpr

    RelayExpr has different format for operators, functions,
        which is hard to apply uniform transformation pass.
        Such as the `TupleGetItem`.

    Abstract representation allows different definitions
        for operators, which can be easier for graph
        transformation. Like the `BatchNorm` op returns
        a 3-tuple, whereas the return is first in cvm.

    We need to consistently print symbol information such as name,
        for the user's config about quantization layers.
    """

    name: str
    op_name: str
    args: typing.List[Symbol]
    attrs: typing.Dict[str, typing.Any]

    # TODO: validate variable name has name_hint attribute.

    def __hash__(self) -> int:
        return hash(str(self))

    def is_op(self, op_name):
        return self.op_name == op_name

    def to_dict(self):
        return dict((f.name, getattr(self, f.name)) \
                for f in fields(self))

    def clone(self, cls: typing.Type[Symbol] = None, **kw):
        cls = cls or type(self)
        assert is_dataclass(cls)

        data = {}
        for k in [f.name for f in fields(cls)]:
            if k in [f.name for f in fields(self)]:
                data[k] = getattr(self, k)
        data.update(kw)

        try:
            new = cls(**data)
        except Exception as e:
            if "params" in data:
                del data["params"]
            print("clone failed: ", cls.__name__,
                    self, "\n{}".format(data))
            raise e
        return new

    @property
    def shape(self) -> ShapeT:
        return list(self.attrs["shape"])

    @property
    def dtype(self):
        return self.attrs["dtype"]

    def __eq__(self, other: Symbol):
        return self.args == other.args and hash(self) == hash(other)

    def __repr__(self) -> str:
        args_info= ["{}@{}".format(
            i.name, i.shape) for i in self.args ]
        return "{} = {}({}) /* attrs */ \t{}".format(
            self.name, self.op_name,
            ", ".join(args_info), self.attrs)

    def raw_str(self) -> str:
        shape = ",".join([str(s) for s in self.shape])
        args_info = "({})".format(
                ", ".join([i.name for i in self.args]))
        skips = [ "shape", "dtype", "name_hint" ]
        attrs = {k: self.attrs[k] \
                for k in self.attrs if k not in skips}
        return "{:30} = {:>15}{:30} /* attrs */ {}".format(
                "{}@({})".format(self.name, shape),
                self.op_name, args_info, attrs or "")


def _topo_sort(symbol: Symbol, sym_list: typing.List[Symbol]):
    if sym_list.count(symbol) > 0:
        return
    for c in symbol.args:
        _topo_sort(c, sym_list)
    sym_list.append(symbol)

_VisitorT = typing.Callable[[Symbol], None]
_TransformerT = typing.Callable[[Symbol], typing.Optional[Symbol]]
""" Symbol Transformer

    Return new symbol to transform old symbol into updated one,
        or just return None for symbol visit.
"""

def visit(symbol: Symbol, callback: _VisitorT):
    """ Visitor mode, possible modify symbol itself. """
    sym_list: typing.List[Symbol] = []
    _topo_sort(symbol, sym_list)
    for sym in sym_list:
        callback(sym)


def transform(symbol: Symbol, callback: _TransformerT) -> Symbol:
    """ Transform symbol from old to new, with inputs updated.

        Only the return value indicates mutation, while changing
        attributes in parameter passed in args does nothing.
    """
    sym_list: typing.List[Symbol] = []
    _topo_sort(symbol, sym_list)

    sym_map = {}
    for sym in sym_list:
        args = [sym_map[c.name] for c in sym.args]
        # pre-clone symbol, to avoid misleading usage in callback
        sym = sym.clone(
                args=args,
                attrs={k: v for k, v in sym.attrs.items()})
        out = callback(sym) or sym
        assert isinstance(out, Symbol)
        sym_map[sym.name] = out
    return sym_map[symbol.name]

def filter_operators(*op_names: typing.List[str]):
    def _pass(f):
        @wraps(f)
        def _wrapper(sym: Symbol, *args, **kw) -> typing.Any:
            if any([ sym.is_op(n) for n in op_names ]):
                return f(sym, *args, **kw)
        return _wrapper
    return _pass

