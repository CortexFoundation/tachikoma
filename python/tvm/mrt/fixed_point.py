
import numpy as np
from dataclasses import dataclass

from . import op
from .opns import *
from .precision import WithPrecision
from .attrs import PClipAttrs, RequantAttrs
from .symbol import filter_operators
from .transform import Transformer
from .quantize import QuantizedInfo

@dataclass(repr=False)
class FixPoint(Transformer, QuantizedInfo):
    def __repr__(self, **attrs):
        attrs.setdefault("tinfer", self.dtype)
        attrs.setdefault("sinfer", self.shape)
        return super().__repr__(**attrs)

    def map_requant(self):
        X: MapRequant = self.args[0]
        parsed: RequantAttrs = self.parsed

        anno_bit = WithPrecision.MAX_BIT / 2
        if X.precision > anno_bit:
            rs_bit = X.from_const_data(X.precision - anno_bit)
            X = op.right_shift(X, rs_bit).like(self)

        frac, exp = cvm_float(self.parsed.rescale, anno_bit)
        assert frac > 1
        frac_sym = X.from_const_data(frac)
        out = op.mul(X, frac_sym).like(self)

        exp_sym = out.from_const_data(exp)
        out = op.right_shift(out, exp_sym)
        return out.like(self)

    def map_pclip(self):
        X: MapRequant = self.args[0]
        parsed: PClipAttrs = self.parsed

        pos = self.int_max()
        out = op.clip(X, a_min=-pos, a_max=pos)
        return out.like(self)

    def set_dtype(self):
        assert 0 <= self.precision and self.precision <= 32
        dtype = "int8" if self.precision <= 8 else "int32"
        self.attrs["dtype"] = dtype

    def __call__(self):
        self.validate_precision()

        self.set_dtype()
        if self.is_input():
            pass
        elif self.is_param():
            data = np.round(self.numpy()).astype("int32")
            absmax = np.abs(data).max()
            assert absmax <= self.int_max()
            self = self.from_np_data(data).like(self)
        elif self.is_op(PCLIP):
            self = self.map_pclip()
        elif self.is_op(REQUANT):
            self = self.map_requant()
        elif self.is_op(CONV2D):
            self.attrs["out_dtype"] = "int32"

        tmp = op.retrieve_operator(self)
        tmp = op.infer_type(tmp)
        assert "int" in tmp.dtype, tmp
        return self

def cvm_float(number, bits=24):
    """ Recalculate the float value within the given range of bits.

        Parameters
        __________
        number : float
            The input float value.
        bits : int
            The target bits to represent the value.

        Returns
        _______
        ret : tuple
            The recalculated float value with its corresponding bits to be shifted.
    """
    alpha = max((2 ** (bits - 1)) - 1, 1)
    bits -= 1
    assert number >= 0
    if number == 0:
        return 0, 0
    exp = 0
    while (number >= 1):
        number /= 2
        exp += 1
    while (number < 1):
        number *= 2
        exp -= 1
    while (bits > 1):
        if (int(number) == number):
            break
        number *= 2
        exp -= 1
        bits -= 1
    frac, sb = round(number), exp
    return min(frac, alpha), sb
