from __future__ import annotations

import typing
from dataclasses import dataclass

import math
import numpy as np

from .opns import *
from .utils import number_to_bits, count_to_bits
from .symbol import Symbol
from .transform import Transformer, Pass

__ALL__ = [ "WithPrecision",
        "InferPrecision", "QuantizedInfo",
]

@dataclass(repr=False)
class WithPrecision(Symbol):
    precision: int

    MAX_BIT: typing.ClassVar[int] = 32

    def __repr__(self, **attrs):
        attrs.setdefault("pinfer", self.precision)
        return super().__repr__(**attrs)

    def validate_precision(self):
        assert isinstance(self.precision, int), self.precision
        assert self.precision <= self.MAX_BIT, (
            "precision:{} out of max bit:{} for \n{}"
        ).format(self.precision, self.MAX_BIT, self)
        assert self.precision > 0

    def int_max(self):
        return (2 ** (self.precision - 1)) - 1

    @classmethod
    def update_dict(cls, data: dict, **kwargs) -> dict:
        prec = data["precision"]
        assert prec is None or prec > 0, prec
        return super().update_dict(data, **kwargs)

@dataclass(repr=False)
class QuantizedInfo(WithPrecision):
    dt_info: str

    def __repr__(self, **attrs):
        attrs["dt_info"] = self.dt_info
        return super().__repr__(**attrs)

    def like(self, other: Symbol, copy=True, **kwargs):
        data = other.to_dict()
        data.update(self.to_dict())
        data["precision"] = self.precision if copy else 0
        data["dt_info"] = self.dt_info if copy else ""
        return type(other).from_dict(data, **kwargs)

@dataclass(repr=False)
class InferPrecision(Pass):
    """ Infer Precision Pass

        This inference should be consistent with cvm-runtime.
    """
    @property
    def arg_precisions(self):
        for a in self.args:
            assert a.precision is not None and a.precision > 0
        return [a.precision for a in self.args]

    def _infer_index(self, index):
        return self.arg_precisions[index]

    def _infer_max(self):
        return max(self.arg_precisions)

    def _infer_mul(self):
        return sum(self.arg_precisions)

    def _first_like(self):
        return self._infer_index(0)

    def _infer_add(self):
        return self._infer_max() + 1

    def _infer_nn(self):
        W = self.args[1]
        add_count = np.product(W.shape[1:])
        add_bits = count_to_bits(add_count)
        return self._infer_mul() + add_bits

# default InferPrecision
@InferPrecision.test(VAR)
def _infer_variable(self: InferPrecision):
    if self.is_input():
        return None
    absmax = np.abs(self.numpy()).max()
    return number_to_bits(absmax)

InferPrecision.test(TUPLE)(InferPrecision._infer_max)
@InferPrecision.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item(self: InferPrecision):
    return self._infer_index(self.parsed.index)
InferPrecision.test(CONV2D, DENSE)(InferPrecision._infer_nn)
InferPrecision.test(BIAS_ADD)(InferPrecision._infer_add)
InferPrecision.test(RELU, MAX_POOL2D)(InferPrecision._first_like)
InferPrecision.test(SQUEEZE, RESHAPE)(InferPrecision._first_like)
@InferPrecision.test(SUM)
def _infer_sum_prec(self: InferPrecision):
    input_len = np.product(self.args[0].shape)
    output_len = np.product(self.shape)
    assert input_len % output_len == 0
    count = int(input_len / output_len)
    sum_bit = count_to_bits(count)
    return self._infer_max() + sum_bit
InferPrecision.test(ADD, SUB)(InferPrecision._infer_add)
InferPrecision.test(MUL)(InferPrecision._infer_mul)
@InferPrecision.test(CLIP)
def _infer_clip(self: InferPrecision):
    a_min = self.attrs["a_min"]
    a_max = self.attrs["a_max"]
    absmax = max(math.fabs(a_min), math.fabs(a_max))
    return number_to_bits(absmax)
@InferPrecision.test(RIGHT_SHIFT)
def _infer_right_shift(self: InferPrecision):
    A, B = self.args[0], self.args[1]
    assert B.is_param()
    b_prec = InferPrecision.bind(B)
    return A.precision - b_prec

def _infer_attr_prec(self: InferPrecision):
    return self.parsed.precision
InferPrecision.test(REQUANT)(_infer_attr_prec)
InferPrecision.test(PCLIP)(_infer_attr_prec)
InferPrecision.test(RS_PCLIP)(_infer_attr_prec)

