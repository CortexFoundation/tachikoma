from __future__ import annotations

import typing
from dataclasses import dataclass, field

from . import op
from .discrete import *
from .precision import *
from .transform import Transformer
from .annotate import ArgAnnotator

__ALL__ = [ "Quantizer" ]

@dataclass(repr=False)
class Quantizer(Transformer, QuantizedInfo):
    """ MRT quantization class.

        Lose the discretor information if dump.
    """
    dt: Discretor | None = field(repr=False)
    revised: Quantizer | None = field(repr=False)
    requants: typing.Dict[str, Quantizer] = field(repr=False)

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        kwargs.setdefault("dt", None)
        kwargs.setdefault("revised", None)
        kwargs.setdefault("requants", {})
        return super().default_dict(**kwargs)

    @classmethod
    def update_dict(cls, data_dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        dt = data_dict.get("dt", None)
        assert dt is None or isinstance(dt, Discretor)
        if dt is not None:
            data_dict["precision"] = dt.precision
            data_dict["dt_info"] = dt.summary()
        return super().update_dict(data_dict)

    def __repr__(self, **attrs):
        # attrs.setdefault("dt", self.dt.summary())
        return super().__repr__(**attrs)

    def __call__(self):
        if self.is_variable():
            return self

        arg_dts: typing.List[Discretor] = ArgAnnotator.bind(self)
        for i in range(len(self.args)):
            arg: Quantizer = self.args[i]
            if arg.revised is not None:
                arg = arg.revised
            self.args[i] = arg.requantize(arg_dts[i])

        self.dt.info = InferDiscretor.bind(self)
        self.dt.precision = InferPrecision.bind(self)
        self = self.copy() # update info
        self.examine_precision()
        # print("[ Quantize]", self)
        return InferOperator.bind(self).like(self)

    def examine_precision(self):
        """ set revised target with discretor examine.
                use explicit clip to annotate precision
                if necessary.
        """
        new_dt: Discretor = self.dt.copy()
        new_dt.examine()
        out = self
        if self.dt.precision > new_dt.precision:
            out = op.pclip(out, precision=new_dt.precision)
            # print("[    PClip]>> {}".format(out))
        self.revised = out.like(self, dt=new_dt)
        # raw_print(self.revised)

    def requantize(self, new_dt: Discretor):
        new_dt.examine()
        key = new_dt.summary()
        if key in self.requants:
            return self.requants[key]

        if self.is_input():
            out = self.copy()
        elif self.is_param():
            out = new_dt.mapping(self)
        else:
            out = new_dt.remapping(self.dt, self)
        out = out.like(self, dt=new_dt)
        # out.is_op(REQUANT) and print("[  Requant]>> {}".format(out))
        self.requants[key] = out
        return out

