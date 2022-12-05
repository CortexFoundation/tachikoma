from __future__ import annotations

import typing
import numpy as np

import tvm

from dataclasses import dataclass, field, InitVar

from .symbol import *
from . import runtime
from .transform import Transformer
from .types import *

@dataclass
class Calibrator(Transformer):
    is_nd: bool = False
    nd_data: typing.List[tvm.nd.NDArray] = field(default_factory=list)
    np_data: typing.List[np.ndarray] = field(default_factory=list)

    def __call__(self,
            data: tvm.nd.NDArray | None =None,
            data_dict: ParametersT = {}):
        if self.is_input():
            out = data_dict.get(self.name, data)
            if out is None:
                # use random input data
                out = np.random.randn(*self.shape)
                out = out.astype(self.dtype)
                out = tvm.nd.array(out)
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

        self.np_data = [ d.numpy() for d in self.nd_data ]

    def run(self, args_data: typing.Dict[str, tvm.nd.NDArray]):
        if self.is_op(TUPLE_GET_ITEM_NAME):
            return self.args[0].flat_nd_data[self.parsed.index]

        args = [ a.as_parameter() for a in self.args]
        sym = self.clone(Symbol, args=args)
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
