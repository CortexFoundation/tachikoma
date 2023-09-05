from __future__ import annotations

import typing
from functools import wraps
from dataclasses import dataclass, field

import numpy as np

import tvm
from tvm import relay, ir

# from .trace import *
from .symbol import *

from . import op, opns
from .attrs import _BaseAttrs, parse_attrs

from .utils import N

@dataclass(repr=False)
class WithParameters(Symbol):
    """ Type TransformerT for Trace """
    parsed: _BaseAttrs = field(repr=False)
    params: ParametersT = field(repr=False)

    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        parsed = parse_attrs(
                data_dict["op_name"], data_dict["attrs"])
        return super().update_dict(data_dict, parsed=parsed)

    def __repr__(self, **attrs):
        if self.is_param():
            attrs["absmax"] = np.abs(self.numpy()).max()
        return super().__repr__(**attrs)

    def ndarray(self) -> tvm.nd.NDArray:
        assert self.is_param(), (
            "{} is not parameter.").format(self.name)
        return self.params[self.name]

    def numpy(self) -> np.ndarray:
        return self.ndarray().numpy()

    def as_parameter(self, data: np.ndarray):
        # TODO: move to symbol
        self.params[self.name] = tvm.nd.array(data.astype(self.dtype))
        return self.copy(op_name=opns.VAR, args=[], attrs={})

    def from_const_data(self, data: typing.Union[int, float]) -> Symbol:
        return self.from_np_data(np.array(data).astype(self.dtype))

    def from_np_data(self, data: np.ndarray, prefix=None) -> Symbol:
        name = N.n(prefix=prefix)
        self.params[name] = tvm.nd.array(data)
        return op.variable(
                name, data.shape, data.dtype.name).like(self)

    def is_input(self) -> bool:
        return op.is_input(self, self.params)
    def is_param(self) -> bool:
        return op.is_param(self, self.params)
    def is_variable(self) -> bool:
        return op.is_variable(self, self.params)
    def is_operator(self) -> bool:
        return op.is_operator(self, self.params)


PassFuncT = typing.Callable[[Symbol], typing.Any]
OpRegistryT = typing.Dict[str, PassFuncT]
ClsRegistryT = typing.Dict[typing.Type, OpRegistryT]

_PASS_REGISTRY: ClsRegistryT = {}
@dataclass(repr=False)
class Pass(WithParameters):
    """ check every operator to be examined in pass. """
    origin: Transformer

    @classmethod
    def _register_op(cls, f, *op_names, callback=None):
        _PASS_REGISTRY.setdefault(cls, {})
        op_registry = _PASS_REGISTRY[cls]
        for opn in op_names:
            callback and callback(opn, op_registry)
            _PASS_REGISTRY[cls][opn] = f

    @classmethod
    def unmount_all(cls):
        _PASS_REGISTRY.setdefault(cls, {})
        _PASS_REGISTRY[cls].clear()

    @classmethod
    def test(cls, *op_names):
        def check_non_override(opn, op_registry):
            assert opn not in op_registry, (
                "Registry for {}:{} is overrided"
            ).format(cls.__name, opn)

        def _func(f, *args, **kw):
            @wraps(f)
            def _wrapper(sym: Symbol):
                return f(sym, *args, **kw)
            cls._register_op(_wrapper, *op_names,
                    callback=check_non_override)
            return f
        return _func

    @classmethod
    def test_all(cls, f, *args, **kw):
        @wraps(f)
        def _wrapper(sym: Symbol):
            return f(sym, *args, **kw)
        cls._register_op(_wrapper, "*")
        return f

    @classmethod
    def replace(cls, *op_names):
        def check_non_override(opn, op_registry):
            assert opn in op_registry, (
                "Registry for {}:{} is not exists"
            ).format(cls.__name, opn)

        def _func(f, *args, **kw):
            @wraps(f)
            def _wrapper(sym: Symbol):
                return f(sym, *args, **kw)
            cls._register_op(_wrapper, *op_names,
                    callback=check_non_override)
            return f
        return _func


    # TODO: add unmount and unmount all function

    @classmethod
    def bind(cls, symbol: Symbol, **kwargs):
        return cls.base(symbol)(**kwargs)

    def __call__(self, *args, **kw):
        op_registry = _PASS_REGISTRY[type(self)]
        for opn, reg in op_registry.items():
            if self.is_op(opn):
                return reg(self, *args, **kw)

        if "*" in op_registry:
            return op_registry["*"](self, *args, **kw)

        assert False, "{} don't supported op:{}".format(
                type(self).__name__, self.op_name)



@dataclass(repr=False)
class Transformer(WithParameters):
    """ Symbol Transformer """

    def to_dict(self, **kwargs):
        """ override to dict, since transformer may want to
                access the previous tfm. Thus, the next
                update_dict has the `origin` key by default.
        """
        return super().to_dict(origin=self, **kwargs)

    @classmethod
    def apply(cls, *args, **kw):
        """ Static apply function to generator transformer pass.

        All the parameters are used to invoke `call` method.
        """
        def _tfm(sym: Symbol, params: ParametersT):
            ins = cls.base(sym, params=params)
            out = ins(*args, **kw) or ins
            assert isinstance(out, cls), (
                "expected {}, but get {}"
                    ).format(cls, type(out))
            return out

        _tfm.__name__ = cls.__name__
        return _tfm

    def __call__(self, *args, **kw) -> Symbol:
        raise NotImplementedError()

class Quantizer(Transformer):
    def expect_max_precision(self, max_prec) -> Quantizer:
        """ Requantization Method  """
        raise NotImplementedError("")

