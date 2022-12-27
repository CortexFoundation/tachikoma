from __future__ import annotations

import typing
from dataclasses import dataclass, field

from .opns import *
from .utils import *
from .calibrate import Sampling
from .precision import WithPrecision, QuantizedInfo
from .transform import Pass, Transformer

__ALL__ = [
        "Discretor",
        "InferPrecision", "InferDiscretor",
        "InferOperator", ]

@dataclass(repr=False)
class Discretor(Sampling, QuantizedInfo):
    """ Perform discretization on the sampling data
            and precision.
    """
    @classmethod
    def update_dict(cls, data: dict, **kwargs):
        cls.update_extra_attrs(
                data, dt_type=get_class_name(cls))
        return super().update_dict(data, **kwargs)

    # ======== Annotate Functions ==========
    def same(self, other: Discretor) -> Discretor:
        """ make current discretization same as other. """
        return self.copy().set_extra_attrs(
            dt_info=other.dt_info, precision=other.precision)

    def set_prec(self, prec: typing.Any) -> Discretor:
        return self.copy().set_extra_attrs(
                dt_info=None, precision=prec)

    # ======== Quantize Functions ==========
    def mapping(self, data: np.ndarray) -> np.ndarray:
        """ discrete parameters. """
        self.examine()
        return self._mapping(data)

    def restore(self, data: np.ndarray) -> np.ndarray:
        """ restore discreted parameters. """
        self.examine()
        return self._restore(data)

    def remapping(self, base: Discretor, sym: Transformer) -> Transformer:
        """ Remapping discretor to another precision. """
        self.examine()
        if self.dt_info == base.dt_info:
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
    def _mapping(self, data):
        raise NotImplementedError()
    def _restore(self, data):
        raise NotImplementedError()
    def _remapping(self, base, sym):
        raise NotImplementedError()
    def _examine(self):
        raise NotImplementedError()

@dataclass(repr=False)
class InferDiscretor(Pass):
    """ Discretization Information Inference with Operator """
    args: typing.List[QuantizedInfo]
    @property
    def arg_infos(self):
        return [a.dt_info for a in self.args]

@dataclass(repr=False)
class InferOperator(Pass):
    """ default operator inference. """
    def identity(self):
        return self

InferOperator.test_all(InferOperator.identity)

