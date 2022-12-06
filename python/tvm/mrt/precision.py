from __future__ import annotations

import typing
from dataclasses import dataclass, field, make_dataclass

import numpy as np

from .symbol import *
from .op import *
from .transform import Transformer, Pass 

def count_to_bits(count: int):
    """
    # get_bit_cnt (mrt) should be consistent with
    # GetReduceSumBit (cvm-runtime)

    """
    prec = 0
    while count != 0:
        prec += 1
        count >>= 1
    return prec


class Precision(int):
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        assert self >= 0
        return self

    def defined(self):
        return self > 0

@dataclass
class WithPrecision(Symbol):
    precision: Precision

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        return super().default_dict(
                **kwargs, precision=Precision())

    def defined(self):
        return self.precision > 0

@dataclass(repr=False)
class Annotate(Pass):
    arg_precisions: typing.List[Precision] = \
            field(default_factory=list)
    """ annotate arg precisions to requant. """

    def set_arg_precision(self, prec: int):
        self.arg_precisions = [
                Precision(prec) for _ in self.args ]

    def identity(self):
        self.set_arg_precision(0)

Annotate.ignore(VAR)

Annotate.test(CONV2D, DENSE)(Annotate.set_arg_precision, 8)
Annotate.test(MUL, ADD, SUB)(Annotate.set_arg_precision, 16)
Annotate.test(TUPLE, TUPLE_GET_ITEM)(Annotate.identity)
Annotate.test(RELU, MAX_POOL2D)(Annotate.identity)
Annotate.test(SQUEEZE, RESHAPE)(Annotate.identity)

@dataclass(repr=False)
class InferPrecision(Pass, WithPrecision):
    """ infered precision as expected. """

    def raw_str(self):
        return super().raw_str(precision=self.precision)

    def _set_prec(self, prec: int):
        self.precision = Precision(prec)

    @property
    def arg_precisions(self):
        assert all([a.precision.defined() for a in self.args])
        return [a.precision for a in self.args]

    def _infer_index(self, index):
        self._set_prec(self.arg_precisions[index])

    def _infer_max(self):
        self._set_prec(max(self.arg_precisions))

    def _infer_mul(self):
        self._set_prec(sum(self.arg_precisions))

    def _first_like(self):
        self._infer_index(0)

    def _infer_add(self):
        self._infer_max()
        self._set_prec(self.precision + 1)

    def _infer_nn(self):
        W = self.args[1]
        add_count = np.product(W.shape[1:])
        add_bits = count_to_bits(add_count)
        self._infer_mul()
        self._set_prec(self.precision + add_bits)


InferPrecision.ignore(VAR)
InferPrecision.test(TUPLE)(InferPrecision._infer_max)
@InferPrecision.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item(self: InferPrecision):
    return self._infer_index(self.parsed.index)
InferPrecision.test(CONV2D, DENSE)(InferPrecision._infer_nn)
InferPrecision.test(BIAS_ADD)(InferPrecision._infer_add)
InferPrecision.test(RELU, MAX_POOL2D)(InferPrecision._first_like)
InferPrecision.test(SQUEEZE, RESHAPE)(InferPrecision._first_like)
InferPrecision.test(ADD, SUB)(InferPrecision._infer_add)
InferPrecision.test(MUL)(InferPrecision._infer_mul)
# InferPrecision.test(REQUANT)(InferPrecision._first_like)

@dataclass(repr=False)
class InferScale(Pass):
    scale: float = 1

    @property
    def arg_scales(self):
        return [a.scale for a in self.args]

    def _infer_index(self, index):
        self.scale = self.arg_scales[index]

    def _first_like(self):
        self._infer_index(0)

    def _uniform_scales(self):
        scales = self.arg_scales
        assert scales.count(scales[0]) == len(scales)
        self._infer_index(0)

    def _infer_mul(self):
        self.scale = np.product(self.arg_scales)


InferScale.ignore(VAR)
InferScale.test(TUPLE)(InferScale._uniform_scales)
@InferScale.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item_scale(self: InferScale):
    return self._infer_index(self.parsed.index)
InferScale.test(CONV2D, DENSE)(InferScale._infer_mul)
InferScale.test(BIAS_ADD)(InferScale._uniform_scales)
InferScale.test(RELU, MAX_POOL2D)(InferScale._first_like)
InferScale.test(SQUEEZE, RESHAPE)(InferScale._first_like)
InferScale.test(ADD, SUB)(InferScale._uniform_scales)
InferScale.test(MUL)(InferScale._infer_mul)
