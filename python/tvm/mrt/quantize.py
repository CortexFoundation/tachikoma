
import typing

from dataclasses import dataclass

from .transform import Transformer
from .calibrate import Calibrator

@dataclass
class Scaler(Symbol):
    args: typing.List[Calibrator]

    precision: int
    """ output precision """

    def __call__(self):
        return self.max_prec(precision)

    def max_prec(self, prec: int) -> Scaler:
        raise NotImplementedError()


@dataclass
class Quantizer(Transformer):
    """ Scale current operator into integer range.

    """
    args: typing.List[Scaler]

    max_bit: int = 32
    """ maximum bit for quantization. """
    #  precision: int = -1
    #  arg_precs: typing.List[int]
    cached_prec_args: typing.Dict[int, Scaler]

    def __call__(self, max_bit: int):
        self.max_bit = max_bit
        self.arg_precs = self.calc_args_prec()
        assert len(arg_precs) == len(self.args)

        args = [ a.max_prec() for a in self.args ]
        return 

    def calc_args_prec(self) -> typing.List[int]:
        raise NotImplementedError()

    def max_prec(self, prec: int) -> Scaler:
        if prec not in self.cached_prec_args:
            self.cached_prec_args[prec] = self._max_prec(prec)
        return self.cached_prec_args[prec]

    def _max_prec(self, prec: int) -> Scaler:
        raise NotImplementedError()
