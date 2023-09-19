from __future__ import annotations

import typing
from dataclasses import dataclass, field

from .opns import *
from .symbol import *
from . import op, utils

@dataclass(repr=False)
class WithScale(Symbol):
    @classmethod
    def _validate(cls, scale, msg=None):
        assert scale >= 0, ("scale: {} invalid for \n{}").format(
                scale, msg or str(cls))

    def __repr__(self, **attrs):
        return super().__repr__(scale=self.scale)

    @property
    def scale(self):
        return self.extra_attrs.get("scale", -1)

    @scale.setter
    def scale(self, val):
        self._validate(val)
        self.set_extra_attrs(scale=val)

    @property
    def scale_defined(self) -> bool:
        return self.scale >= 0

ScaleRulesT = typing.Callable[[WithScale], typing.Any]
_INFER_SCALE_RULES: typing.Dict[str, ScaleRulesT] = {}

def register_scale_rules(*op_names, rule: ScaleRulesT = None):
    assert rule is not None
    for op in op_names:
        _INFER_SCALE_RULES[op] = rule

def scale_rules(*op_names):
    def _add_rules(f: ScaleRulesT):
        for op in op_names:
            _INFER_SCALE_RULES[op] = f
        return f
    return _add_rules

def scale_index(s: WithScale, index: int):
    return s.args[index].scale

def scale_nn(s: WithScale):
    return s.args[0].scale * s.args[1].scale

def scale_identity(s: WithScale):
    return s.args[0].scale

def infer_scale(symbol: WithScale):
    def _infer(sym: Symbol):
        sym = WithScale.base(sym)
        if op.is_variable(sym):
            assert sym.scale_defined, ("var: %s cannot deduct scale"
                    ) % sym.name
            return
        assert sym.op_name in _INFER_SCALE_RULES, (
                "infer scale not support for op:%s"
                ) % sym.op_name
        sym.scale = _INFER_SCALE_RULES[sym.op_name](sym)
        return sym
    out: WithScale = transform(symbol, _infer)
    return out.scale

