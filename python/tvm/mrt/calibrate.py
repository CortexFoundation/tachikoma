from __future__ import annotations

import typing
import numpy as np

import tvm

from dataclasses import dataclass, field, InitVar

from .symbol import *
from . import op
from .sym_expr import *
from . import runtime
from .transform import Transformer
from .types import *

@dataclass(repr=False)
class Calibrator(Transformer):
    is_nd: bool = False
    nd_data: typing.List[tvm.nd.NDArray] = field(
            repr=False, default_factory=list)
    """ not to dump, and restore from np_data. """
    data: typing.List[np.ndarray] = field(default_factory=list)

    @classmethod
    def update_dict(cls, data_dict, **kwargs):
        np_data = data_dict.get("data", None)
        nd_data = data_dict.get("nd_data", None)
        if np_data is not None and nd_data is None:
            nd_data = [ tvm.nd.array(d) for d in np_data ]
            data_dict["nd_data"] = nd_data
        return super().update_dict(data_dict, **kwargs)

    def _rand_data(self,
            enabled: bool = False,
            absmax: float | None = None,
    ):
        assert enabled, "symbol:{} don't have data".format(
                self.name)
        out = np.random.randn(*self.shape)
        out = out.astype(self.dtype)
        if absmax is not None:
            assert absmax > 0
            norm = np.abs(out).max()
            out = out * absmax / norm
        return tvm.nd.array(out)

    def __call__(self,
            data: tvm.nd.NDArray | None = None,
            data_dict: ParametersT = {},
            random_config: typing.Dict[str, typing.Any] = {},
    ):
        if self.is_input():
            out = data_dict.get(self.name, data)
            out = out or self._rand_data(**random_config)
        elif self.is_param():
            out = self.params[self.name]
        else:
            out = self.run({ a.name: a.flat_nd_data \
                    for a in self.args })

        assert isinstance(out, (tvm.nd.NDArray, list)), type(out)
        if isinstance(out, tvm.nd.NDArray):
            self.is_nd = True
            self.nd_data = [ out, ]
            self._assert(out.dtype, self.dtype)
            self._assert(out.shape, self.shape)
        else:
            self.is_nd = False
            self.nd_data = out
            self._assert([o.dtype for o in out], self.dtype)
            self._assert([o.shape for o in out], self.shape)

        self.data = [ d.numpy() for d in self.nd_data ]

    def run(self, args_data: typing.Dict[str, tvm.nd.NDArray]):
        if self.is_op(TUPLE_GET_ITEM):
            return self.args[0].nd_data[self.parsed.index]
        elif self.is_op(REQUANT):
            return self.args[0].flat_nd_data

        sym: Symbol = op.retrieve_operator(self)
        expr = symbol2expr(sym)
        return runtime.infer(expr, args_data)

    @property
    def flat_nd_data(self):
        return self.nd_data[0] if self.is_nd else self.nd_data

    def _assert(self, val, expect):
        if isinstance(val, (list, tuple)):
            assert len(val) == len(expect), (
                    "{} vs. {}").format(val, expect)
            for v, e in zip(val, expect):
                self._assert(v, e)
            return
        assert val == expect, "{} vs. {}".format(val, expect)


@dataclass(repr=False)
class Sampling(Transformer):
    @property
    def data(self) -> typing.Any:
        return self.extra_attrs["data"]

    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        origin = data_dict.get("origin", None)
        if isinstance(origin, Calibrator):
            data = cls.sampling(origin.data)
            cls.update_extra_attrs(data_dict, data=data)
        return super().update_dict(data_dict)

    @classmethod
    def sampling(cls, np_data: np.ndarray) -> typing.Any:
        raise NotImplementedError()

    def __call__(self, *args, **kw):
        return self

@dataclass(repr=False)
class SymmetricMinMaxSampling(Sampling):
    @property
    def data(self) -> float:
        return super().data

    @classmethod
    def sampling(cls, data: np.ndarray) -> float:
        if isinstance(data, list):
            return max([cls.sampling(d) for d in data])
        return float(np.abs(data).max())


