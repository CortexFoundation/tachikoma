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

def scale_rules(*op_names):
    def _add_rules(f: ScaleRulesT):
        for op in op_names:
            _INFER_SCALE_RULES[op] = f
        return f
    return _add_rules

def _scale_index(s: WithScale, index: int):
    return s.args[index].scale

@scale_rules(TUPLE)
def _scale_tuple(s: WithScale):
    return [a.scale for a in s.args]

@scale_rules(TUPLE_GET_ITEM)
def _scale_tuple_get_item(s: WithScale):
    return s.args[0].scale[s.parsed.index]

@scale_rules(CONV2D, DENSE, MUL)
def _scale_nn(s: WithScale):
    return s.args[0].scale * s.args[1].scale

@scale_rules(REQUANT, PCLIP, RS_PCLIP)
@scale_rules(SUM,  RIGHT_SHIFT)
@scale_rules(CLIP, SQUEEZE, RESHAPE, RELU, MAX_POOL2D)
@scale_rules(ADD, SUB, BIAS_ADD)
def _scale_identity(s: WithScale):
    return s.args[0].scale

def infer_scale(symbol: WithScale):
    def _infer(sym: Symbol):
        sym = WithScale.base(sym)
        if op.is_variable(sym):
            assert sym.scale_defined, ("var: %s cannot deduct scale"
                    ) % sym.name
            return
        assert sym.op_name in _INFER_SCALE_RULES, (
                "infer scale error for unknown op:%s"
                ) % sym.op_name
        sym.scale = _INFER_SCALE_RULES[sym.op_name](sym)
        return sym
    out: WithScale = transform(symbol, _infer)
    return out.scale

