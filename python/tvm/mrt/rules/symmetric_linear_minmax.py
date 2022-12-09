from __future__ import annotations

from dataclasses import dataclass

from ..opns import *
from ..utils import *

from .. import op
from ..transform import Pass
from ..discrete import Discretor, InferDiscretor
from ..calibrate import SymmetricMinMaxSampling

@dataclass(repr=False)
class SymmetricLinearDiscretor(
        Discretor, SymmetricMinMaxSampling):
    info: float | None
    """ symmetric linear scale to precision integer. """
    precision: int | None

    @classmethod
    def update_dict(cls, data_dict, **kwargs) -> dict:
        if "origin" in data_dict:
            assert isinstance(
                data_dict["origin"], SymmetricMinMaxSampling), \
                type(data_dict["origin"])
        assert isinstance(data_dict["data"], float), \
                type(data_dict["data"])
        return super().update_dict(data_dict, **kwargs)

    def summary(self) -> str:
        return "T({:.3f})|S({:.2f})|P({})".format(
                self.data or 0, self.info or 0,
                self.precision or 0)

    def _mapping(self, sym: Symbol) -> Quantizer:
        out: Quantizer = sym.copy(name=N.n())
        out.update_data(sym.numpy() * self.info)

        # params out of precision will be cliped
        #   in cvm-runtime.
        check = self.sampling(out.numpy())
        checked_bit = number_to_bits(check)
        assert checked_bit <= self.precision
        return out

    def _remapping(self,
            base: SymmetricLinearDiscretor,
            sym: Symbol) -> Quantizer:
        rescale = self.info / base.info
        out = op.requant(sym,
                rescale=rescale,
                precision=self.precision)
        return out

    def _examine(self):
        if self.info is not None:
            real_max = self.data * self.info
            self.precision = number_to_bits(real_max)
        elif self.precision is not None:
            prec_max = 2 ** (self.precision - 1) - 1
            self.info = prec_max / self.data
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


