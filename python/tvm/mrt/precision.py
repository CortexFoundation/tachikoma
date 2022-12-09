from __future__ import annotations

import typing
from dataclasses import dataclass

from .symbol import Symbol

__ALL__ = [ "WithPrecision" ]

@dataclass(repr=False)
class WithPrecision(Symbol):
    precision: int | None

    MAX_BIT: typing.ClassVar[int] = 32

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        return super().default_dict(precision=None, **kwargs)

    def __repr__(self, **attrs):
        if self.precision is not None:
            attrs.setdefault("precision", self.precision)
        return super().__repr__(**attrs)

    def validate_precision(self):
        if self.precision is None:
            return
        assert self.precision <= self.MAX_BIT, (
            "precision:{} out of max bit:{} for \n{}"
        ).format(self.precision, self.MAX_BIT, self)
        assert self.precision > 0

    @classmethod
    def update_dict(cls, data: dict, **kwargs) -> dict:
        prec = data["precision"]
        assert prec is None or prec > 0
        return super().update_dict(data, **kwargs)
