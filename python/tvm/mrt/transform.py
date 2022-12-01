from __future__ import annotations

import typing
from dataclasses import dataclass, fields

import tvm
from tvm import relay, ir

from . import transformers
from .symbol import *
from .trace import *

@dataclass
class Transformer(Symbol):
    """ Type TransformerT for Trace """

    params: ParametersT = field(default_factory=dict)

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

