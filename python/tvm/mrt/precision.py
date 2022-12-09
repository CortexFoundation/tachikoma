from __future__ import annotations

import typing
from dataclasses import dataclass

from .symbol import Symbol

__ALL__ = [ "WithPrecision" ]

@dataclass(repr=False)
class WithPrecision(Symbol):
    precision: int

    MAX_BIT: typing.ClassVar[int] = 32

    def __repr__(self, **attrs):
        attrs.setdefault("pinfer", self.precision)
        return super().__repr__(**attrs)

    def validate_precision(self):
        assert isinstance(self.precision, int)
        assert self.precision <= self.MAX_BIT, (
            "precision:{} out of max bit:{} for \n{}"
        ).format(self.precision, self.MAX_BIT, self)
        assert self.precision > 0

    def int_max(self):
        return (2 ** (self.precision - 1)) - 1

    @classmethod
    def update_dict(cls, data: dict, **kwargs) -> dict:
        prec = data["precision"]
        assert prec is None or prec > 0, prec
        return super().update_dict(data, **kwargs)
