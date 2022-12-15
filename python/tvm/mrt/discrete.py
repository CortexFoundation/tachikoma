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
        kwargs.setdefault("info", None)
        return super().default_dict(**kwargs)

    def to_dict(self, **kwargs) -> dict:
        kwargs.setdefault("dt", self)
        return super().to_dict(**kwargs)

    # def __repr__(self, **attrs):
    #     if self.info is not None:
    #         attrs.setdefault("discrete_info", self.info)
    #     return super().__repr__(**attrs)

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
class InferDiscretor(Pass):
    """ Discretization Information Inference with Operator """
    @property
    def arg_infos(self):
        return [a.dt.info for a in self.args]

@dataclass(repr=False)
class InferOperator(Pass):
    """ default operator inference. """
    def identity(self):
        return self

InferOperator.test_all(InferOperator.identity)

