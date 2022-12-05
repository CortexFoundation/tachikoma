import math
import typing

from dataclasses import dataclass, field, InitVar

from .symbol import *
from .transform import Transformer
from .calibrate import Calibrator

#  @dataclass
#  class Scaler:
#      sym: InitVar[Calibrator]

#      data: typing.Any = field(init=False)
#      """ basic data information scaler must know. """
#      scale: typing.Any = 1
#      precision: int = -1

#      def max_prec(self, prec: int) -> Scaler:
#          raise NotImplementedError()

#      def at_scale(self, scale: float) -> Scaler:
#          return self

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

class Scaler:
    pass

@dataclass
class SymmetricMinMaxScaler(Scaler):
    sym: InitVar[Calibrator]
    data: float = field(init=False)
    """ threshold for calib data. """

    scale: typing.Any = 1
    precision: int = 0

    def __post_init__(self, sym: Calibrator):
        self.data = sym.np_data.abs().max().scalar()
        if self.precision > 0:
            prec_max = 2 ** (self.precision - 1) - 1
            self.scale = self.prec_max / self.data
        elif self.scale != 1:
            real_max = self.data * self.scale
            self.precision = number_to_bits(real_max)


@dataclass(repr=False)
class Quantizer(Transformer):
    """ Scale current operator into integer range.

    """
    origin: Symbol
    """ original symbol data. """

    max_bit: int = 32
    """ maximum bit for quantization. """
    #  precision: int = -1
    #  arg_precs: typing.List[int]
    cached_prec_args: typing.Dict[int, Scaler] = field(default=dict)

    @classmethod
    def base(cls, symbol: Symbol, **kwargs):
        return cls.from_dict(
                symbol.to_dict(**kwargs),
                origin=symbol)

    def __call__(self, max_bit: int):
        self.max_bit = max_bit
        self.arg_precs = self.calc_args_prec()
        assert len(arg_precs) == len(self.args)

        args = [ a.max_prec() for a in self.args ]
        return self

    def calc_args_prec(self) -> typing.List[int]:
        raise NotImplementedError()

    def max_prec(self, prec: int) -> Scaler:
        if prec not in self.cached_prec_args:
            self.cached_prec_args[prec] = self._max_prec(prec)
        return self.cached_prec_args[prec]

    def _max_prec(self, prec: int) -> Scaler:
        raise NotImplementedError()
