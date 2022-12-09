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
        return "(scale){},(precision){}".format(
                self.info, self.precision)

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

@dataclass(repr=False)
class SymmetricLinearInferDiscretor(Pass):
    def infer_index(self, index):
        return self.arg_infos[index]

    def first_like(self):
        infos = self.arg_infos
        assert infos.count(infos[0]) == len(infos), infos
        return self.infer_index(0)

    def infer_mul(self):
        """ only SymmetricLinearDiscretor """
        return np.product(self.arg_infos)

# set default InferDiscretor for symmetric linear discretor.
InferDiscretor.test(VAR)(lambda x: None)
InferDiscretor.test(TUPLE)(InferDiscretor.first_like)
@InferDiscretor.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item_scale(self: InferDiscretor):
    return self._infer_index(self.parsed.index)
InferDiscretor.test(CONV2D, DENSE)(InferDiscretor.infer_mul)
InferDiscretor.test(BIAS_ADD)(InferDiscretor.first_like)
InferDiscretor.test(RELU, MAX_POOL2D)(InferDiscretor.first_like)
InferDiscretor.test(SUM)(InferDiscretor.first_like)
InferDiscretor.test(SQUEEZE, RESHAPE)(InferDiscretor.first_like)
InferDiscretor.test(ADD, SUB)(InferDiscretor.first_like)
InferDiscretor.test(MUL)(InferDiscretor.infer_mul)


