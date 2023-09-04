from __future__ import annotations

import typing
from dataclasses import dataclass, field

from .opns import *
from . import op
from .symbol import *
from .utils import *
from .calibrate import Sampling
from .precision import WithPrecision, QuantizedInfo, infer_precision
from .scaler import WithScale, infer_scale
from .transform import Pass, Transformer

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
            out = op.requant(
                    self, rescale=scale / curr_scale,
                    precision=precision
                    ).like(self)
            out.set_extra_attrs(
                    data=self.data, scale=scale, precision=precision)
            self.requant_ops[info] = out
        return self.requant_ops[info]

RequantRulesT = typing.Callable[[QuantInfo], DiscreteInfo]
""" Returns expected scale and precision.

    None, None indicate that none operation to requant.
"""
_DISCRETE_REQUANT_RULES: typing.Dict[str, RequantRulesT] = {}
OpRulesT = typing.Callable[[QuantInfo], Symbol]
_DISCRETE_OP_RULES: typing.Dict[str, OpRulesT] = {}

def requant_rules(*op_names):
    def _add_rules(f: RequantRulesT):
        for op in op_names:
            _DISCRETE_REQUANT_RULES[op] = f
        return f
    return _add_rules

def op_rules(*op_names):
    def _add_rules(f: OpRulesT):
        for op in op_names:
            _DISCRETE_OP_RULES[op] = f
        return f
    return _add_rules

@requant_rules(MAX_POOL2D)
@requant_rules(RELU, CLIP, RESHAPE, SQUEEZE)
def _identity(s: QuantInfo):
    return [DiscreteInfo() for c in s.args]

def syms_prec(syms: typing.List[QuantInfo], prec: int):
    return [DiscreteInfo(precision=prec) for s in syms]
def args_prec(s: QuantInfo, prec: int):
    return syms_prec(s.args, prec)

requant_rules(VAR)(lambda s: 0)
requant_rules(CONV2D, DENSE)(lambda s: args_prec(s, 8))
requant_rules(MUL)(lambda s: args_prec(s, 8))
requant_rules(SUM)(lambda s: args_prec(s, 10))

@requant_rules(ADD, SUB, BIAS_ADD)
def add_sub_rules(s: QuantInfo):
    std_prec = 15
    # standard max precision for add/sub children.

    assert len(s.args) == 2
    A: QuantInfo = s.args[0]
    B: QuantInfo = s.args[1]
    assert A.is_operator() or B.is_operator(), "need fuse constant"
    if A.scale_defined and A.precision < std_prec:
        scaleA = A.scale
    else:
        scaleA = A.precision_to_scale(std_prec)

    if B.scale_defined and B.precision < std_prec:
        scaleB = B.scale
    else:
        scaleB = B.precision_to_scale(std_prec)

    scale = min(scaleA, scaleB)
    return [DiscreteInfo(scale=scale) for c in s.args]

@op_rules(CLIP)
def _op_clip(s: QuantInfo):
    scale = s.args[0].scale
    s.set_extra_attrs(
            a_min=s.parsed.a_min * scale,
            a_max=s.parsed.a_max * scale)
    return s.copy()


@dataclass(repr=False)
class Discretor2(QuantInfo):
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

        arg_dts = _DISCRETE_REQUANT_RULES[self.op_name](self)
        print("arg dts:", [(a.scale, a.precision) for a in arg_dts])
        for i, arg in enumerate(self.args):
            self.args[i] = arg.rescale(arg_dts[i])
            print(self.args[i])

        out = self
        if self.op_name in _DISCRETE_OP_RULES:
            out = _DISCRETE_OP_RULES[self.op_name](out).like(self)

        new = op.subgraph(out, inames=[a.name for a in self.args])
        #  raw_print(new)
        scale = infer_scale(new)
        precision = self.scale_to_precision(scale)

        out = op.pclip(out, precision=precision).like(out)
        out.set_extra_attrs(
                data=self.data, scale=scale, precision=precision)
        raw_print(op.subgraph(out, inames=orig_names))
        return out


@dataclass(repr=False)
class Discretor(Sampling, QuantizedInfo):
    """ Perform discretization on the sampling data
            and precision.
    """
    @classmethod
    def update_dict(cls, data: dict, **kwargs):
        cls.update_extra_attrs(
                data, dt_type=get_class_name(cls))
        return super().update_dict(data, **kwargs)

    # ======== Annotate Functions ==========
    def same(self, other: Discretor) -> Discretor:
        """ make current discretization same as other. """
        return self.copy().set_extra_attrs(
            dt_info=other.dt_info, precision=other.precision)

    def set_prec(self, prec: typing.Any) -> Discretor:
        return self.copy().set_extra_attrs(
                dt_info=None, precision=prec)

    # ======== Quantize Functions ==========
    def mapping(self, data: np.ndarray) -> np.ndarray:
        """ discrete parameters. """
        self.examine()
        return self._mapping(data)

    def restore(self, data: np.ndarray) -> np.ndarray:
        """ restore discreted parameters. """
        self.examine()
        return self._restore(data)

    def remapping(self, base: Discretor, sym: Transformer) -> Transformer:
        """ Remapping discretor to another precision. """
        self.examine()
        if self.dt_info == base.dt_info:
            return sym
        return self._remapping(base, sym)

    def examine(self):
        """ Use sampling data to revise discretor information.
        """
        self.validate_precision()
        self._examine()
        self.validate_precision()

    def summary(self) -> str:
        """ return current discrete information. """
        raise NotImplementedError()
    def _mapping(self, data):
        raise NotImplementedError()
    def _restore(self, data):
        raise NotImplementedError()
    def _remapping(self, base, sym):
        raise NotImplementedError()
    def _examine(self):
        raise NotImplementedError()

@dataclass(repr=False)
class InferDiscretor(Pass):
    """ Discretization Information Inference with Operator """
    args: typing.List[QuantizedInfo]
    @property
    def arg_infos(self):
        return [a.dt_info for a in self.args]

@dataclass(repr=False)
class InferOperator(Pass):
    """ default operator inference. """
    def identity(self):
        return self

InferOperator.test_all(InferOperator.identity)

