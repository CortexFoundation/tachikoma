from __future__ import annotations

from dataclasses import dataclass

from ..opns import *
from ..utils import *

from .. import op
from ..transform import Pass, Transformer
from ..discrete import Discretor, InferDiscretor

@dataclass(repr=False)
class SymmetricLinearDiscretor(Discretor):
    """ symmetric linear scale to precision integer. """

    @property
    def data(self) -> float:
        return super().data
    @property
    def dt_info(self) -> float:
        return super().dt_info
    @dt_info.setter
    def dt_info(self, val):
        self.set_extra_attrs(dt_info=val)

    @classmethod
    def update_dict(cls, data_dict, **kwargs) -> dict:
        data = data_dict["extra_attrs"]["data"]
        assert isinstance(data, float), type(data)
        return super().update_dict(data_dict, **kwargs)

    def summary(self) -> str:
        return "T({:.3f})|S({:.2f})".format(
                self.data, self.dt_info or 0)

    def _mapping(self, data: np.ndarray) -> np.ndarray:
        out = data * self.dt_info
        # params out of precision will be cliped
        #   in cvm-runtime.
        check = float(np.abs(out).max())
        checked_bit = number_to_bits(check)
        if checked_bit > self.precision:
            print((
                "[WARNING]: precision is out of bound"
                ", expected {}, but get {}.").format(
                    self.precision, checked_bit))
        return out.astype("float32")

    def _restore(self, data: np.ndarray) -> np.ndarray:
        out = data / self.dt_info
        return out.astype("float32")

    def _remapping(self,
            base: SymmetricLinearDiscretor,
            sym: Symbol) -> Quantizer:
        rescale = self.dt_info / base.dt_info
        out = op.requant(sym,
                rescale=rescale,
                precision=self.precision)
        return out.like(sym)

    def _examine(self):
        if self.dt_info is not None:
            real_max = self.data * self.dt_info
            self.precision = number_to_bits(real_max)
        elif self.precision is not None:
            prec_max = 2 ** (self.precision - 1) - 1
            self.dt_info = prec_max / self.data
        else:
            assert False

# set default InferDiscretor for symmetric linear discretor.
""" SymmetricLinearInferDiscretor  """
def slid_infer_index(self: InferDiscretor, index):
    return self.arg_infos[index]
def slid_first_like(self: InferDiscretor):
    infos = self.arg_infos
    assert infos.count(infos[0]) == len(infos), infos
    return slid_infer_index(self, 0)
def slid_infer_mul(self: InferDiscretor):
    """ only SymmetricLinearDiscretor """
    return np.product(self.arg_infos)

InferDiscretor.unmount_all()
InferDiscretor.test(VAR)(lambda x: None)
InferDiscretor.test(TUPLE)(slid_first_like)
@InferDiscretor.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item_scale(self: InferDiscretor):
    return slid_infer_index(self.parsed.index)
InferDiscretor.test(CONV2D, DENSE)(slid_infer_mul)
InferDiscretor.test(BIAS_ADD)(slid_first_like)
InferDiscretor.test(RELU, MAX_POOL2D)(slid_first_like)
InferDiscretor.test(SUM)(slid_first_like)
InferDiscretor.test(SQUEEZE, RESHAPE)(slid_first_like)
InferDiscretor.test(ADD, SUB)(slid_first_like)
InferDiscretor.test(MUL)(slid_infer_mul)


