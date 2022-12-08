from __future__ import annotations

import math
import typing

from dataclasses import dataclass, field, InitVar, asdict

from .utils import *
from .symbol import *
from . import op
from .transform import Transformer
from .calibrate import Calibrator
from .scale import *
from .precision import *

@dataclass(repr=False)
class Quantizer(Transformer):
    """ Scale current operator into integer range.

    """
    scaler: Scaler

    # MAX_BIT: typing.ClassVar[int] = 32
    """ maximum bit for quantization. """

    tight_precision: Quantizer | None
    requants: typing.Dict[int, Quantizer]

    @property
    def scale(self):
        return self.scaler.scale
    @property
    def precision(self):
        return self.scaler.precision

    @classmethod
    def default_dict(cls) -> dict:
        return super().default_dict(
            tight_precision = None,
            requants = {})

    def __repr__(self):
        return super().__repr__(
                precision=self.precision,
                scale=self.scale)

    @classmethod
    def base(cls, symbol: Scaler, **kwargs):
        assert isinstance(symbol, Scaler), type(symbol)
        return cls.from_dict(
                symbol.to_dict(**kwargs),
                scaler=symbol)

    def __call__(self):
        scalers: AnnotateT = Annotate.bind(self)

        for i in range(len(self.args)):
            arg  = self.args[i].tight_precision
            self.args[i] = arg.requantize(scalers[i])
            print("arg: ", i, self.args[i])


        ip = InferPrecision.bind(self)
        sc = InferScale.bind(self)
        self.scaler.set(sc, ip)
        # print(ip.raw_str(), sc.scale)
        # self = Rewriter.base(self)()

        self.examine_precision()
        # print("> [quantized]", self)
        return self

    def examine_precision(self):
        new_scaler: Scaler = self.scaler.copy()
        new_scaler.examine()

        out = self
        if self.scaler.precision > new_scaler.precision:
            out = op.pclip(out, precision=new_scaler.precision)
        self.tight_precision = out.like(self, scaler=new_scaler)

    def requantize(self, new_scaler: Scaler) -> Requantizer:
        """ if set scale, use scale to calculate prec. """
        # new_scaler: Scaler = self.scaler.copy()
        # new_scaler.examine(scale, prec)
        key = new_scaler.hash()
        if key in self.requants:
            return self.requants[key]

        out = new_scaler.rescale(self.scaler, self)
        out = out.like(self, scaler=new_scaler)
        self.requants[key] = out
        return out

    def calc_args_prec(self) -> typing.List[int]:
        raise NotImplementedError()

    def max_prec(self, prec: int) -> Scaler:
        if prec not in self.cached_prec_args:
            self.cached_prec_args[prec] = self._max_prec(prec)
        return self.cached_prec_args[prec]

    def _max_prec(self, prec: int) -> Scaler:
        raise NotImplementedError()
