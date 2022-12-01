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
    args: typing.List[Calibrator]

    is_nd: bool = False
    output: typing.List[np.ndarray] = field(default_factory=list)

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
        elif self.is_op(TUPLE_GET_ITEM_NAME):
            out = self.args[0].raw_output[self.attrs["index"]]
            assert isinstance(out, tvm.nd.NDArray), type(out)
        else:
            out = self.run({ a.name: a.raw_output \
                    for a in self.args })

        if isinstance(out, tvm.nd.NDArray):
            self.is_nd = True
            self.output = [ out, ]
            self._assert(out.dtype, self.dtype)
            self._assert(out.shape, self.shape)
        else:
            self.is_nd = False
            self.output = out
            self._assert([o.dtype for o in out], self.dtype)
            self._assert([o.shape for o in out], self.shape)

        print(self.name, self.op_name, self.shape, self.dtype)

    def run(self, args_data: typing.Dict[str, tvm.nd.NDArray]):
        args = [ a.as_parameter() for a in self.args]
        sym = self.clone(Symbol, args=args)
        expr = symbol2expr(sym)
        #  data = { a.name: a.raw_output for a in self.args }
        return runtime.infer(expr, args_data)

    @property
    def raw_output(self):
        return self.output[0] if self.is_nd else self.output

    def _assert(self, val, expect):
        if isinstance(val, (list, tuple)):
            assert len(val) == len(expect), (
                    "{} vs. {}").format(val, expect)
            for v, e in zip(val, expect):
                self._assert(v, e)
            return
        assert val == expect, "{} vs. {}".format(val, expect)

    #  @property
    #  def shape(self):
    #      return self.output[0].shape if self.is_nd \
    #              else [ o.shape for o in self.output ]

    #  @property
    #  def dtype(self):
    #      return self.output[0].shape if self.is_nd \
    #              else [ o.shape for o in self.output ]
