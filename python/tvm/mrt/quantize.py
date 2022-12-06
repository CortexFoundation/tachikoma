from __future__ import annotations

import math
import typing

from dataclasses import dataclass, field, InitVar, asdict

from .symbol import *
from .transform import Transformer
from .calibrate import Calibrator
from .precision import *

def number_to_bits(number: float):
    """ Return the integer bits to represent number.
        precision bit: 1
        number bits:
            [ 0-0 ] => 0, skip
            [ 1-1 ] => 1, ceil(log2(i+1)) = 1
            [ 2-3 ] => 2, ceil(log2(i+1)) = 2
            [ 4-7 ] => 3, ceil(log2(i+1)) = 3
            ...

        return 1 + ceil(log2(number + 1))

        note: consider the abs round int for number.
    """
    number = math.fabs(number)
    number = math.floor(number + 0.5)
    return 1 + math.ceil(math.log2(number + 1))

class Requantizer(Transformer):
    def __call__(self, expected: Precision):
        if self.is_input():
            return self.requantize_input(expected)
        elif self.is_param():
            return self.requantize_param()
        self.requantize_operator()

    # def requantize_input(self):
    #     return Requantizer.


@dataclass(repr=False)
class Quantizer(Transformer, WithPrecision):
    """ Scale current operator into integer range.

    """
    # scaler: Scaler

    MAX_BIT: typing.ClassVar[int] = 32
    """ maximum bit for quantization. """

    tight_precision: Quantizer | None = None
    cached_requant: typing.Dict[int, Scaler] = field(default=dict)

    # @property
    # def scale(self):
    #     return selc.scaler.scale

    # @classmethod
    # def base(cls, symbol: Calibrator, **kwargs):
    #     assert isinstance(sym, Calibrator)
    #     scaler = SymmetricMinMaxScaler.base(symbol)
    #     return cls.from_dict(
    #             symbol.to_dict(**kwargs),
    #             scaler=SymmetricMinMaxScaler.base(symbol))

    def __call__(self,
            scaler_type: typing.Type[Scaler] = SymmetricMinMaxScaler):
        anno: Annotate = Annotate.base(self)

        for i in range(len(self.args)):
            expected = anno.arg_precisions[i]
            self.args[i] = self.args[i].tight_precision
            self.args[i] = self.args[i].requantize(expected)

        ip = InferPrecision.base(self)
        sc = InferScale.base(self)
        print(ip.raw_str(), sc.scale)
        # self = Rewriter.base(self)()
        self.require_tight_prec(scaler_type(sc.scale))

        return self

    def require_tight_prec(self, scale):
        self.tight_precision = self.copy(
            scaler=self.scaler.copy(scale=scale))

    def requantize(self):
        return self

    def calc_args_prec(self) -> typing.List[int]:
        raise NotImplementedError()

    def max_prec(self, prec: int) -> Scaler:
        if prec not in self.cached_prec_args:
            self.cached_prec_args[prec] = self._max_prec(prec)
        return self.cached_prec_args[prec]

    def _max_prec(self, prec: int) -> Scaler:
        raise NotImplementedError()
