from __future__ import annotations

import typing
from dataclasses import dataclass, field

from . import op
from .discrete import *
from .transform import Transformer
from .annotate import ArgAnnotator

__ALL__ = [ "Quantizer" ]

@dataclass(repr=False)
class Quantizer(Transformer):
    """ MRT quantization class.

        Lose the discretor information if dump.
    """
    summary: str | None
    dt: Discretor | None = field(repr=False)
    revised: Quantizer | None = field(repr=False)
    requants: typing.Dict[str, Quantizer] = field(repr=False)

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        return super().default_dict(
                dt = None, summary=None,
                revised = None,
                requants = {}, **kwargs)

    def __repr__(self, **attrs):
        attrs.setdefault("summary", self.summary)
        return super().__repr__(**attrs)

    @classmethod
    def update_dict(cls, data_dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        dt = data_dict.get("dt", None)
        if dt is None and "origin" in data_dict:
            dt : Discretor = data_dict["origin"]
            assert isinstance(dt, Discretor)
        if dt is not None:
            data_dict["summary"] = dt.summary()
        return super().update_dict(data_dict, dt=dt)

    def __call__(self):
        arg_dts: typing.List[Discretor] = ArgAnnotator.bind(self)

        for i in range(len(self.args)):
            arg: Quantizer = self.args[i].revised
            self.args[i] = arg.requantize(arg_dts[i])

        self.revised = self
        if self.is_operator():
            self.dt.info = InferDiscretor.bind(self)
            self.dt.precision = InferPrecision.bind(self)
            self.summary = self.dt.summary()
            self.examine_precision()
        return InferOperator.bind(self)

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
        self.requants[key] = out
        return out

