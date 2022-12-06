from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np

import tvm
from tvm import relay, ir

from . import transformers, op
from .symbol import *
from .trace import *
from .attrs import _BaseAttrs, parse_attrs

from .utils import N

@dataclass(repr=False)
class WithParameters(Symbol):
    """ Type TransformerT for Trace """
    parsed: _BaseAttrs = field(repr=False)
    params: ParametersT = field(repr=False)

    @classmethod
    def update_dict(cls, data: dict) -> dict:
        return super().update_dict(data,
            parsed=parse_attrs(data["op_name"], data["attrs"]))

    def ndarray(self) -> tvm.nd.NDArray:
        assert self.is_param(), (
            "{} is not parameter.").format(self.name)
        return self.params[self.name]

    def numpy(self) -> np.ndarray:
        return self.ndarray().numpy()

    def update_data(self, data: np.ndarray):
        self.params[self.name] = tvm.nd.array(data)

    def from_np_data(self,
            data: np.ndarray,
            prefix=None,
    ) -> Transformer:
        new_name = N.n(prefix=prefix)
        self.params[new_name] = tvm.nd.array(data)
        return op.variable(new_name,
                data.shape, data.dtype.name).like(self)

    def is_input(self) -> bool:
        return is_input(self, self.params)
    def is_param(self) -> bool:
        return is_param(self, self.params)
    def is_variable(self) -> bool:
        return is_variable(self, self.params)
    def is_operator(self) -> bool:
        return is_operator(self, self.params)


@dataclass(repr=False)
class Transformer(WithParameters):
    """ Type TransformerT for Trace """
    @classmethod
    def apply(cls, *args, **kw):
        def _tfm(sym: Symbol, params: ParametersT):
            ins = cls.base(sym, params=params)
            out = ins(*args, **kw) or ins
            return out.like(ins)

        _tfm.__name__ = cls.__name__
        return _tfm

    def __call__(self, *args, **kw) -> Symbol:
        raise NotImplementedError()

PassFuncT = typing.Callable[[Symbol], typing.Any]

@dataclass(repr=False)
class Pass(WithParameters):
    """ check every operator to be examined in pass. """
    OP_REGISTRY: typing.ClassVar[typing.Dict[str, PassFuncT]] = {}

    @classmethod
    def test(cls, *op_names):
        def _func(f, *args, **kw):
            def _wrapper(sym: Symbol):
                return f(sym, *args, **kw)
            for opn in op_names:
                cls.OP_REGISTRY[opn] = _wrapper
            return f
        return _func

    @typing.final
    def __post_init__(self):
        for opn, reg in self.OP_REGISTRY.items():
            if self.is_op(opn):
                reg(self)
                return

        assert False, "{} don't supported op:{}".format(
                type(self), self.op_name)


    @staticmethod
    def _pass_identity(*args, **kw):
        pass

    @classmethod
    def ignore(cls, *op_names):
        for opn in op_names:
            cls.OP_REGISTRY[opn] = cls._pass_identity
        return cls._pass_identity

class Quantizer(Transformer):
    def expect_max_precision(self, max_prec) -> Quantizer:
        """ Requantization Method  """
        raise NotImplementedError("")

