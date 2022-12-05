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
class Transformer(Symbol):
    """ Type TransformerT for Trace """
    parsed: _BaseAttrs
    params: ParametersT

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

    @classmethod
    def apply(cls, *args, **kw):
        def _tfm(sym: Symbol, params: ParametersT):
            ins = cls.base(sym, params=params,
                    parsed=parse_attrs(sym.op_name, sym.attrs))
            out = ins(*args, **kw) or ins
            return out.like(ins)

        _tfm.__name__ = cls.__name__
        return _tfm

    def __call__(self, *args, **kw) -> Symbol:
        return self


class Validator(Transformer):
    pass

class Quantizer(Transformer):
    def expect_max_precision(self, max_prec) -> Quantizer:
        """ Requantization Method  """
        raise NotImplementedError("")

