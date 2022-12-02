from __future__ import annotations

import typing
from dataclasses import dataclass, fields

import numpy as np

import tvm
from tvm import relay, ir

from . import transformers
from .symbol import *
from .trace import *
from .attrs import _BaseAttrs, parse_attrs

from .utils import N

@dataclass
class Transformer(Symbol):
    """ Type TransformerT for Trace """

    args: typing.List[Transformer]
    params: ParametersT
    parsed: typing.Type[_BaseAttrs | None] = fields(init=False)

    def __post_init__(self):
        self.parsed = parse_attrs(self.op_name, self.attrs)

    def ndarray(self) -> tvm.nd.NDArray:
        assert self.is_param(), (
            "{} is not parameter.").format(self.name)
        return self.params[self.name]

    def numpy(self) -> np.ndarray:
        return self.ndarray().numpy()

    def update_data(self, data: np.ndarray):
        self.params[self.name] = tvm.nd.array(data)

    def from_np_data(self, data: np.ndarray) -> Transformer:
        new_name = N.n()
        self.params[new_name] = tvm.nd.array(data)
        return Transformer(name, VAR_NAME, [],
                {
                    "shape": data.shape,
                    "dtype": data.dtype,
                    "name_hint": new_name,
                }, self.params)

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
        def _tfm(symbol: Symbol, params: ParametersT):
            ins = symbol.clone(cls, params=params)
            return ins(*args, **kw) or ins
        return _tfm

    def __call__(self, *args, **kw) -> Symbol:
        return self


class Validator(Transformer):
    pass

class Quantizer(Transformer):
    def expect_max_precision(self, max_prec) -> Quantizer:
        """ Requantization Method  """
        raise NotImplementedError("")

