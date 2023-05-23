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

def _format_printer(data):
    if isinstance(data, dict):
        data = ["{}={}".format(k, _format_printer(v)) \
                for k, v in data.items()]
        return ", ".join(data)
    elif isinstance(data, (tuple, list)):
        return "(" + ",".join([_format_printer(d) \
                for d in data]) + ")"
    elif isinstance(data, float):
        return "{:.3f}".format(data)
    return str(data)[-20:]

@dataclass(repr=False)
class _BaseSymbol:
    """ Symbol should record neccessary infomation about
            the transformer, such as discretization method,
            precision, etc.
    """
    extra_attrs: typing.Dict[str, typing.Any]
    """ extra attributes will be inherited automatically. """

    @classmethod
    def update_extra_attrs(cls, data_dict, **kwargs):
        extra_attrs: dict = data_dict.get("extra_attrs", {})
        extra_attrs.update(kwargs)
        data_dict["extra_attrs"] = extra_attrs
        return data_dict
    def set_extra_attrs(self, **kwargs) -> Symbol:
        self.extra_attrs.update(kwargs)
        return self

    @classmethod
    def base(cls, symbol: Symbol, **kwargs):
        """ create current class instance based on another.
            Enable the inherit class to override.
        """
        return cls.from_dict(symbol.to_dict(), **kwargs)
    def like(self, other: Symbol, **kwargs) -> Symbol:
        """ cast current symbol to child class. """
        data = other.to_dict()
        data.update(self.to_dict())
        return type(other).from_dict(data, **kwargs)
    def copy(self, **kwargs) -> typing.Type[Symbol]:
        """ clone current symbol. """
        return type(self).from_dict(
            self.to_dict(), **kwargs) # kwargs override self

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        """ possible dict to initialize symbol class. """
        kwargs.setdefault("extra_attrs", {})
        return kwargs
    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        return data_dict
    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        fnames = [f.name for f in fields(cls)]
        data = {k: data[k] for k in data if k in fnames}
        try:
            out = cls(**data)
        except Exception as e:
            print(cls, list(data.keys()))
            raise e
        return out
    def to_dict(self, **kwargs) -> dict:
        data = dataclass_to_dict(self)
        data.update(**kwargs)
        data["args"] = [a for a in data["args"]]
        data["attrs"] = {k: v for k, v in self.attrs.items()}
        data["extra_attrs"] = {k: v \
                for k, v in data["extra_attrs"].items()}
        return data

    def __repr__(self, **attrs) -> str:
        args_info = "({})".format(
                ", ".join([i.name for i in self.args]))
        oattrs = {k: v for k, v in self.attrs.items()}
        oattrs.update(attrs)
        oattrs.update(self.extra_attrs)
        return "{:30} = {:>15}{:30}<args_info> /* attrs */ {}".format(
                self.name, self.op_name, args_info,
                _format_printer(oattrs))


@dataclass
class Symbol(_BaseSymbol):
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

    # Overridable Methods, inheritted from _BaseSymbol
    #   to support multi-inherit design.
    @classmethod
    def update_extra_attrs(cls, data_dict, **kwargs):
        return super().update_extra_attrs(data_dict, **kwargs)
    def set_extra_attrs(self, **kwargs):
        return super().set_extra_attrs(**kwargs)
    @classmethod
    def base(cls, symbol: Symbol, **kwargs) -> Symbol:
        return super().base(symbol, **kwargs)
    def like(self, other: Symbol, **kwargs) -> Symbol:
        return super().like(other, **kwargs)
    def copy(self, **kwargs) -> Symbol:
        return super().copy(**kwargs)
    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return super().from_dict(d, **kwargs)
    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        kwargs.setdefault("args", [])
        kwargs.setdefault("attrs", {})
        return super().default_dict(**kwargs)
    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        return super().update_dict(data_dict, **kwargs)
    def to_dict(self, **kwargs) -> dict:
        return super().to_dict(**kwargs)
    def __repr__(self, **attrs) -> str:
        return super().__repr__(**attrs)
    def info(self, **attrs) -> str:
        return super().__repr__(**attrs)
#        inputs_info = [
#            "{}@{}".format(i.name, i.attrs.get("shape", None)) \
#            for i in self.args]
#        return "{} = {}({}) /* attrs */ \t{}".format(
#            self.name, self.op_name,
#            ", ".join(inputs_info),
#            self.attrs)

    # Naive Methods
    def is_op(self, *op_names) -> bool:
        return self.op_name in op_names

    @property
    def shape(self) -> ShapeT:
        return list(self.extra_attrs.get("shape", None))
    @shape.setter
    def shape(self, val):
        self.extra_attrs["shape"] = list(val)
    @property
    def dtype(self):
        return self.extra_attrs.get("dtype", None)
    @dtype.setter
    def dtype(self, val):
        self.extra_attrs["dtype"] = val

    def __hash__(self) -> int:
        return hash(str(self))

def _topo_sort(symbol: Symbol, sym_list: typing.List[Symbol]):
    if sym_list.count(symbol) > 0:
        return
    for c in symbol.args:
        _topo_sort(c, sym_list)
    sym_list.append(symbol)

def sym2list(symbol: Symbol) -> typing.List[Symbol]:
    sym_list = []
    _topo_sort(symbol, sym_list)
    return sym_list

_SymbolNodesT = typing.List[typing.Dict[str, typing.Any]]
_SymbolJsonT = typing.Dict[str, typing.Any]


def dump_json(symbol: Symbol) -> _SymbolJsonT:
    nodes = []
    def _to_json(sym: Symbol):
        node = dataclass_to_dict(sym, check_repr=True)
        node.update({
            "args": [a.name for a in node["args"]],
            "_class_type": get_class_name(sym),
            })
        nodes.append(node)
    visit(symbol, _to_json)
    return { "nodes": nodes, }

def load_json(data: _SymbolJsonT, **extra_attrs) -> Symbol:
    nodes: _SymbolNodesT = data["nodes"]

    sym_map = {}
    for node in nodes:
        args = [sym_map[a] for a in node["args"]]
        sym_type: typing.Type[Symbol] = eval(node["_class_type"])
        sym = sym_type.from_dict(node, args=args, **extra_attrs)
        sym_map[sym.name] = sym
    return sym_map[nodes[-1]["name"]]

_VisitorT = typing.Callable[[Symbol], None]
_TransformerT = typing.Callable[[Symbol], typing.Optional[Symbol]]
""" Symbol Transformer

    Return new symbol to transform old symbol into updated one,
        or just return None for symbol visit.
"""

def visit(symbol: Symbol, callback: _VisitorT):
    """ Visitor mode, possible modify symbol itself. """
    for sym in sym2list(symbol):
        callback(sym)

def transform(symbol: Symbol, callback: _TransformerT) -> Symbol:
    """ Transform symbol from old to new, with inputs updated.

        Only the return value indicates mutation, while changing
        attributes in parameter passed in args does nothing.
    """
    sym_map = {}
    for sym in sym2list(symbol):
        args = [sym_map[c.name] for c in sym.args]
        # pre-clone symbol, to avoid misleading usage in callback
        sym = sym.copy(args=args)
        out = callback(sym) or sym
        assert isinstance(out, Symbol)
        # default const_ prefix symbol means parameters
        assert sym.name not in sym_map, sym.name
        # assert sym.name.startswith("const_") or \
        #         sym.name not in sym_map, sym.name
        sym_map[sym.name] = out
    return sym_map[symbol.name]

def raw_print(symbol: Symbol):
    msg = "{f} Raw Print {f}".format(f = "="*25)
    print(msg)
    def _print(sym: Symbol):
        print(sym)
    visit(symbol, _print)
    print("=" * len(msg))

def filter_operators(*op_names: typing.List[str]):
    def _pass(f):
        @wraps(f)
        def _wrapper(sym: Symbol, *args, **kw) -> typing.Any:
            if any([ sym.is_op(n) for n in op_names ]):
                return f(sym, *args, **kw)
        return _wrapper
    return _pass
