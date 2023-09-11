from __future__ import annotations

import typing
from dataclasses import dataclass, field

from .opns import *
from . import op
from .symbol import *
from .utils import *
from .calibrate import Sampling
from .precision import WithPrecision, QuantizedInfo, infer_precision
from .scaler import *
from .transform import Transformer

__ALL__ = [
        "Discretor",
        "InferPrecision", "InferDiscretor",
        "InferOperator", ]


@dataclass(repr=False, unsafe_hash=True)
class DiscreteInfo:
    scale: typing.Optional[typing.Any] = None
    precision: typing.Optional[int] = None

    @property
    def undefined(self) -> bool:
        return self.scale is None and self.precision is None


@dataclass(repr=False)
class QuantInfo(WithScale, WithPrecision, Sampling):
    requant_ops: typing.Dict[DiscreteInfo, Symbol] = field(repr=False)

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        kwargs.setdefault("requant_ops", {})
        return super().default_dict(**kwargs)

    def scale_to_precision(self, scale):
        real_max = scale * self.data
        return number_to_bits(real_max)
    def precision_to_scale(self, precision):
        return bits_to_number(precision) / self.data

    def rescale(self, info: DiscreteInfo):
        """ rescale current symbol into other, aka requant.

            if scale specified:
                precision = bit(scale * data)
            if precision specified:
                scale = real(precision) / data
            TODO: choose min if both specified
        """
        scale, precision = info.scale, info.precision
        if info.undefined:
            return self
        elif scale is not None:
            precision = self.scale_to_precision(scale)
        elif precision is not None:
            scale = self.precision_to_scale(precision)

        if info not in self.requant_ops:
            curr_scale = self.scale if self.scale_defined else 1
            #TODO: add pass to check rescale=1 and duplicate requant
            out = op.requant(
                    self,
                    rescale=scale/curr_scale,
                    precision=precision,
                    ).like(self)
            out.set_extra_attrs(
                data=self.data, scale=scale, precision=precision)
            self.requant_ops[info] = out
        return self.requant_ops[info]

RequantRulesT = typing.Callable[[QuantInfo], typing.List[DiscreteInfo]]
""" Returns expected scale and precision.

    None, None indicate that none operation to requant.
"""
_DISCRETE_REQUANT_RULES: typing.Dict[str, RequantRulesT] = {}
OpRulesT = typing.Callable[[QuantInfo], Symbol]
_DISCRETE_OP_RULES: typing.Dict[str, OpRulesT] = {}

_requant_identity = lambda s: [DiscreteInfo() for _ in s.args]
_op_identity = lambda s: s

def register_rules(*op_names,
        requant_rule: RequantRulesT | None = None,
        op_rule: OpRulesT | None = None,
        scale_rule: ScaleRulesT | None = None):
    for op in op_names:
        if requant_rule is not None:
            _DISCRETE_REQUANT_RULES[op] = requant_rule
        if op_rule is not None:
            _DISCRETE_OP_RULES[op] = op_rule
        if scale_rule is not None:
            register_scale_rules(op, rule=scale_rule)

def register_rules_with_default(*op_names,
        requant_rule: RequantRulesT | None = None,
        op_rule: OpRulesT | None = None,
        scale_rule: ScaleRulesT | None = None):
    return register_rules(*op_names,
            requant_rule=requant_rule or _requant_identity,
            op_rule=op_rule or _op_identity,
            scale_rule=scale_rule or scale_identity)

def args_max_prec(prec: int):
    def _rule(s: QuantInfo):
        return [DiscreteInfo(precision=prec) for _ in s.args]
    return _rule

register_rules_with_default(
        CONV2D, DENSE, MUL,
        requant_rule=args_max_prec(8),
        scale_rule=scale_nn)
register_rules_with_default(SUM, requant_rule=args_max_prec(10))

def uniform_arg_scales(s: QuantInfo):
    std_prec = 15
    # standard max precision for add/sub children.

    assert len(s.args) > 0
    #  raw_print(s)
    assert any([c.is_operator() for c in s.args]), "Need fuse constant: %s" % s
    scales = []
    for arg in s.args:
        if arg.scale_defined and arg.precision < std_prec:
            scale = arg.scale
        else:
            scale = arg.precision_to_scale(std_prec)
        scales.append(scale)

    target_scale = min(scales)
    return [DiscreteInfo(scale=target_scale) for c in s.args]

#  def uniform_add_sub_scales(s: QuantInfo):
#      assert len(s.args) == 2
#      A: QuantInfo = s.args[0]
#      B: QuantInfo = s.args[1]
#      assert A.is_operator() or B.is_operator(), "need fuse constant"
#      if A.scale_defined and A.precision < std_prec:
#          scaleA = A.scale
#      else:
#          scaleA = A.precision_to_scale(std_prec)

#      if B.scale_defined and B.precision < std_prec:
#          scaleB = B.scale
#      else:
#          scaleB = B.precision_to_scale(std_prec)

#      scale = min(scaleA, scaleB)
#      return [DiscreteInfo(scale=scale) for c in s.args]

register_rules_with_default(
        ADD, SUB, BIAS_ADD, requant_rule=uniform_arg_scales)
register_rules_with_default(
        CONCAT, requant_rule=uniform_arg_scales)

register_rules_with_default(
        MAX_POOL2D, RELU, RESHAPE, SQUEEZE)

def op_clip_rules(s: QuantInfo):
    scale = s.args[0].scale
    s.set_extra_attrs(
            a_min=s.parsed.a_min * scale,
            a_max=s.parsed.a_max * scale)
    return s.copy()

register_rules_with_default(CLIP, op_rule=op_clip_rules)

@dataclass(repr=False)
class Discretor(QuantInfo):
    """
        does operation -> out

        input scales -> output scale
            -> output tighter precision
        # sampling * output scale -> output precision
        input precisions -> output precision
        if output tighter precision < output precision:
            out <- pclip(out, output tighter precision)
            output precision <- output tighter precision

        Case 1: sampling, precision(target) -> scale
        if output precision <= precision:
            scale <- output scale
            precision <- output precision
        else:
            out = requant(out, scale / output scale)
            output precision <- precision(target)
            output scale <- scale

        Case 2: sampling, scale -> precision(target)
        out = requant(out, scale / output scale)
        output precision <- precision(target)
        output scale <- scale
    """
    def __call__(self):
        if self.is_variable():
            return

        orig_names = [a.name for a in self.args]

        assert self.op_name in _DISCRETE_REQUANT_RULES, (
                "requant rules not support for op:{}"
                ).format(self.op_name)
        assert self.op_name in _DISCRETE_OP_RULES, (
                "op rewrite rules not support for op:{}"
                ).format(self.op_name)

        arg_dts = _DISCRETE_REQUANT_RULES[self.op_name](self)
        for i, arg in enumerate(self.args):
            self.args[i] = arg.rescale(arg_dts[i])

        out = _DISCRETE_OP_RULES[self.op_name](self).like(self)

        new = op.subgraph(out, inames=[a.name for a in self.args])
        #  raw_print(new)
        out.scale = infer_scale(new)
        out.precision = self.scale_to_precision(out.scale)

        out = op.pclip(out, precision=out.precision).like(
                out, extra_attrs=out.extra_attrs)
        #  raw_print(op.subgraph(out, inames=orig_names))
        return out

