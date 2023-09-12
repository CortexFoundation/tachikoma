from __future__ import annotations

import typing
import numpy as np

import tvm

from dataclasses import dataclass, field, InitVar

from .types import *
from .symbol import *
from .sym_expr import *
from . import runtime
from . import op, opns, inference
from .transform import Transformer

@dataclass(repr=False)
class Calibrator(Transformer):
    """ not to dump, and restore from np_data. """
    nd_data: OpOutputT | None = field(repr=False, default=None)
    data: np.ndarray | list | None = field(default=None)

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
            device: tvm.runtime.Device = tvm.runtime.cpu(),
            target: tvm.target.Target = tvm.target.arm_cpu(),
    ):
        if self.is_input():
            out = data_dict.get(self.name, data)
            if out is None:
                out = self._rand_data(**random_config)
        elif self.is_param():
            out = self.params[self.name]
        else:
            sym = op.retrieve_operator(self)
            out = inference.run(
                    sym, [ a.nd_data for a in self.args ],
                    device=device, target=target)

        self.nd_data = out
        assert isinstance(out, (tvm.nd.NDArray, list)), type(out)
        if isinstance(out, tvm.nd.NDArray):
            self._assert(out.dtype, self.dtype)
            self._assert(out.shape, self.shape)
            self.data = out.numpy()
        else:
            self._assert([o.dtype for o in out], self.dtype)
            self._assert([o.shape for o in out], self.shape)
            self.data = [ o.numpy() for o in out ]

    def sampling(self, data):
        if isinstance(data, list):
            return max([self.sampling(d) for d in data])
        return 0 if data is None else np.abs(data).max()

    def __repr__(self, **attrs):
        return super().__repr__(
                data=self.sampling(self.data), **attrs)

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
        return self.extra_attrs.get("data", None)
    @data.setter
    def data(self, val):
        self.set_extra_attrs(data=val)

    def __repr__(self, **attrs):
        return super().__repr__(data=self.data, **attrs)

    @classmethod
    def update_dict(cls, data_dict: dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        #  print("data:", data_dict["extra_attrs"]["data"])
        origin = data_dict.get("origin", None)
        if isinstance(origin, Calibrator):
            data = cls.sampling(origin.data)
            assert data > 0
            cls.update_extra_attrs(data_dict, data=data)
        return super().update_dict(data_dict)

    @classmethod
    def sampling(cls, np_data: np.ndarray) -> typing.Any:
        raise NotImplementedError()

    def __call__(self, *args, **kw):
        if self.is_op(CLIP):
            a_min, a_max = self.parsed.a_min, self.parsed.a_max
            self.extra_attrs["data"] = max(abs(a_min), abs(a_max))
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
        data = float(np.abs(data).max())
        assert data > 0
        return data


