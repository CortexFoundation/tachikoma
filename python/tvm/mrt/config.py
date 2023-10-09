from __future__ import annotations

import typing
from dataclasses import dataclass, field

from . import utils

T = typing.TypeVar("T")

class _BaseConfig:
    _last_scope: typing.Optional[_BaseConfig] = None
    __GLOBAL__: typing.ClassVar[_BaseConfig | None] = None

    def __init__(self):
        self._last_scope = self.__GLOBAL__

    @classmethod
    def G(cls: typing.Type[T]) -> T:
        return cls.__GLOBAL__ or cls()

    @classmethod
    def _set_scope(cls: typing.Type[T], ins) -> T:
        cls.__GLOBAL__ = ins
        return ins

    def __enter__(self: typing.Type[T]) -> T:
        return self._set_scope(self)

    def __exit__(self: typing.Type[T], *args) -> T:
        return self._set_scope(self._last_scope)

    def register_global(self: typing.Type[T]) -> T:
        return self._set_scope(self)

    def mutate(self: typing.Type[T], **new_attrs) -> T:
        attrs = utils.dataclass_to_dict(self)
        attrs.update(new_attrs)
        return type(self)(**attrs)

    def restore(self: typing.Type[T], **new_attrs) -> T:
        last_scope = self._last_scope or type(self)()
        return last_scope.mutate(**new_attrs)

@dataclass
class Pass(_BaseConfig):
    name        : str   = ""
    inherit     : bool  = False
    """ whether to inherit config in iterate pass. """
    log_before  : bool  = False
    log_after   : bool  = False

