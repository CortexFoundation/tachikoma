from __future__ import annotations

from dataclasses import InitVar

from .symbol import *
from .trace import *
from . import runtime

@dataclass
class Calibrator(Symbol):
    args: typing.List[Calibrator]

    init_data: InitVar[np.ndarray | None] = None

    is_nd: bool = field(init=False)
    output: typing.List[np.ndarray] = field(init=False)

    def __post_init__(self, init_data):
        if is_variable(self):
            assert isinstance(init_data, tvm.nd.NDArray), type(init_data)
            out = init_data
        elif self.is_op(TUPLE_GET_ITEM_NAME):
            out = self.args[0].raw_output[self.attrs["index"]]
            assert isinstance(out, tvm.nd.NDArray), type(out)
        else:
            out = self.run({ a.name: a.raw_output \
                    for a in self.args })

        if isinstance(out, tvm.nd.NDArray):
            self.is_nd = True
            self.output = [ out, ]
            assert out.dtype == self.dtype, (
                    "{} vs. {}").format(out.dtype, self.dtype)
            assert list(out.shape) == list(self.shape), (
                "{} vs. {}").format(out.shape, self.shape)
        else:
            self.is_nd = False
            self.output = [ o for o in out ]
            assert [o.dtype for o in out] == self.attrs["dtype"]
            assert [o.shape for o in out] == self.attrs["shape"]

        print(self.name, self.op_name, self.shape, self.dtype)

    def run(self, args_data: typing.Dict[str, tvm.nd.NDArray]):
        args = [ a.as_parameter() for a in self.args]
        sym = self.clone(Symbol, args=args)
        expr = symbol2expr(sym)
        data = { a.name: a.raw_output for a in self.args }
        return runtime.infer(expr, data)

    @property
    def raw_output(self):
        return self.output[0] if self.is_nd else self.output

    def _type_assert(self, val, expect):
        if isinstance(val, (list, tuple)):
            assert len(val) == len(expect)
            for v, e in zip(val, expect):
                self._type_assert(v, e)
        assert val == expect, "{} vs. {}".format(val, expect)

    #  @property
    #  def shape(self):
    #      return self.output[0].shape if self.is_nd \
    #              else [ o.shape for o in self.output ]

    #  @property
    #  def dtype(self):
    #      return self.output[0].shape if self.is_nd \
    #              else [ o.shape for o in self.output ]
