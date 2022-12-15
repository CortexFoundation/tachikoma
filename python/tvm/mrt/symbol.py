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

    def is_op(self, *op_names) -> bool:
        return self.op_name in op_names

    def like(self, other: Symbol, **kwargs) -> Symbol:
        """ cast current symbol to child class. """
        data = other.to_dict()
        data.update(self.to_dict())
        return type(other).from_dict(data, **kwargs)

    @classmethod
    def base(cls, symbol: Symbol, **kwargs):
        """ create current class instance based on another.

            Enable the inherit class to override.
        """
        return cls.from_dict(symbol.to_dict(), **kwargs)

    def copy(self, **kwargs) -> typing.Type[Symbol]:
        """ clone current symbol. """
        return type(self).from_dict(
                self.to_dict( # update mutable types
                    args=[a for a in self.args],
                    attrs={k: v for k, v in self.attrs.items()}),
                **kwargs) # kwargs override self

    def to_dict(self, **kwargs) -> dict:
        data = dataclass_to_dict(self)
        data.update(**kwargs)
        return data

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

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        """ possible dict to initialize symbol class. """
        return kwargs

    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        return data_dict

    @property
    def shape(self) -> ShapeT:
        return list(self.attrs.get("shape", None))

    @property
    def dtype(self):
        return self.attrs.get("dtype", None)

    def __repr__(self, **attrs) -> str:
        args_info = "({})".format(
                ", ".join([i.name for i in self.args]))
        attrs.update(self.attrs)
        skips = [ "shape", "dtype", "name_hint" ]
        attrs = {k: attrs[k] for k in attrs if k not in skips}
        return "{:30} = {:>15}{:30} /* attrs */ {}".format(
                self.name, self.op_name, args_info, attrs)

    def __hash__(self) -> int:
        return hash(str(self))

def _topo_sort(symbol: Symbol, sym_list: typing.List[Symbol]):
    if sym_list.count(symbol) > 0:
        return
    for c in symbol.args:
        _topo_sort(c, sym_list)
    sym_list.append(symbol)

_SymbolNodesT = typing.List[typing.Dict[str, typing.Any]]
_SymbolJsonT = typing.Dict[str, typing.Any]

def _class_name(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__


def dump_json(symbol: Symbol) -> _SymbolJsonT:
    nodes = []
    def _to_json(sym: Symbol):
        node = dataclass_to_dict(sym, check_repr=True)
        node.update({
            "args": [a.name for a in node["args"]],
            "_class_type": _class_name(sym),
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
        sym = sym.copy(args=args)
        out = callback(sym) or sym
        assert isinstance(out, Symbol)
        # default const_ prefix symbol means parameters
        assert sym.name.startswith("const_") or \
                sym.name not in sym_map, sym.name
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
