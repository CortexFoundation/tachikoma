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

from .types import *
from .utils import N

@dataclass(repr=False)
class WithParameters(Symbol):
    parsed: _BaseAttrs = field(repr=False)
    params: ParametersT = field(repr=False)
    """ Parameters should not be changed in transformer,
            use copy mode instead to avoid possible errors.
    """

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

    def as_parameter(self, data: OpOutputT):
        # TODO: move to symbol
        assert isinstance(data, tvm.nd.NDArray)
        self.params[self.name] = tvm.nd.array(
                data.numpy().astype(self.dtype))
        return op.as_variable(self)
        #  return self.copy(op_name=opns.VAR, args=[], attrs={})

    def from_const_data(self, data: typing.Union[int, float]) -> Symbol:
        return self.from_np_data(np.array(data))

    def from_np_data(self, data: np.ndarray, prefix=None) -> Symbol:
        name = N.n(prefix=prefix)
        self.params[name] = tvm.nd.array(data.astype(self.dtype))
        return op.variable(
                name, data.shape, self.dtype).like(self)

    def is_input(self) -> bool:
        return op.is_input(self, self.params)
    def is_param(self) -> bool:
        return op.is_param(self, self.params)
    def is_variable(self) -> bool:
        return op.is_variable(self, self.params)
    def is_operator(self) -> bool:
        return op.is_operator(self, self.params)

TransformerT = typing.Callable[[Symbol, ParametersT], Symbol]
""" Transformer Callback Function Type,
        inherited from WithParameters.
"""

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
    def get_transformer(cls, name: typing.Optional[str] = None):
        name = name or cls.__name__
        def _func(symbol: Symbol, params: ParametersT, **kwargs):
            def _run(sym: Symbol):
                out = cls.base(sym, params=params)
                out = out(**kwargs) or out
                assert isinstance(out, cls), (
                        "transform output type should be {}, "
                        "but get {}"
                        ).format(cls, type(out))
                return out
            with N(name):
                return transform(symbol, _run)
        _func.__name__ = name
        return _func

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

    def __call__(self, *args, **kw) -> typing.Optional[Transformer]:
        raise NotImplementedError()

