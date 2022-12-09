from __future__ import annotations

import typing
from dataclasses import dataclass, field

from .opns import *
from .utils import *
from .calibrate import Sampling
from .precision import WithPrecision
from .transform import Pass, Transformer

__ALL__ = [
        "Discretor",
        "InferPrecision", "InferDiscretor",
        "InferOperator", ]

@dataclass(repr=False)
class Discretor(Sampling, WithPrecision):
    """ Perform discretization on the sampling data
            and precision.
    """
    info: typing.Any | None
    """ discretization information
            need to provide __eq__ function to compare
            in InferDiscretor.
    """
    precision: typing.Any | None
    """ precision information
            need to provide base arthmetic function
            to compare in InferPrecision.
    """

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        return super().default_dict(
                info=None, _checked=False, **kwargs)

    def __repr__(self, **attrs):
        if self.info is not None:
            attrs.setdefault("discrete_info", self.info)
        return super().__repr__(**attrs)

    # ======== Annotate Functions ==========
    def same(self, other: Discretor) -> Discretor:
        """ make current discretization same as other. """
        return self.copy(
                info=other.info,
                precision=other.precision)

    def set_prec(self, prec: typing.Any) -> Discretor:
        return self.copy(info=None, precision=prec)

    # ======== Quantize Functions ==========
    def mapping(self, sym: Transformer) -> Transformer:
        """ discrete parameters. """
        self.examine()
        return self._mapping(sym)

    def remapping(self, base: Discretor, sym: Transformer) -> Transformer:
        self.examine()
        if self.info == base.info:
            return sym
        return self._remapping(base, sym)

    def examine(self):
        """ Use sampling data to revise discretor information.
        """
        self.validate_precision()
        self._examine()
        self.validate_precision()

    def summary(self) -> str:
        """ return current discrete information. """
        raise NotImplementedError()
    def _mapping(self, sym):
        raise NotImplementedError()
    def _remapping(self, base, sym):
        raise NotImplementedError()
    def _examine(self):
        raise NotImplementedError()

@dataclass(repr=False)
class InferPrecision(Pass):
    @property
    def arg_precisions(self):
        return [a.dt.precision for a in self.args]

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

InferPrecision.test(VAR)(lambda x: None)
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
# InferPrecision.test(REQUANT)(InferPrecision._first_like)

@dataclass(repr=False)
class InferDiscretor(Pass):
    @property
    def arg_infos(self):
        return [a.dt.info for a in self.args]

    def infer_index(self, index):
        return self.arg_infos[index]

    def first_like(self):
        infos = self.arg_infos
        assert infos.count(infos[0]) == len(infos), infos
        return self.infer_index(0)

    def infer_mul(self):
        """ only SymmetricLinearDiscretor """
        return np.product(self.arg_infos)

@dataclass(repr=False)
class InferOperator(Pass):
    """ default operator inference. """
    def identity(self):
        return self

InferOperator.test_all(InferOperator.identity)

