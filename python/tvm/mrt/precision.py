from __future__ import annotations

import typing
from dataclasses import dataclass

import math
import numpy as np

from .opns import *
from .utils import number_to_bits, count_to_bits, bits_to_number
from .symbol import Symbol, visit, transform
from .transform import Transformer, Pass

__ALL__ = [ "WithPrecision",
        "InferPrecision", "QuantizedInfo",
]

@dataclass(repr=False)
class WithPrecision(Symbol):
    MAX_BIT: typing.ClassVar[int] = 32

    @classmethod
    def _validate(cls, prec, msg=None):
        assert isinstance(prec, int), self.precision
        assert prec <= cls.MAX_BIT, (
            "precision:{} out of max bit:{} for \n{}"
        ).format(prec, cls.MAX_BIT, msg or str(cls))
        assert prec > 0, msg
        return True

    def __repr__(self, **attrs):
        return super().__repr__(prec=self.precision, **attrs)

    @property
    def precision(self) -> int:
        return self.extra_attrs.get("precision", -1)
    @precision.setter
    def precision(self, val):
        self._validate(val, str(self))
        self.set_extra_attrs(precision=val)

    @property
    def defined(self) -> bool:
        """ Whether current precision is well-defined. """
        return self.precision > 0 and self.precision < self.MAX_BIT

    def validate_precision(self):
        self._validate(self.precision, msg=str(self))
    def int_max(self):
        return bits_to_number(self.precision)

@dataclass(repr=False)
class QuantizedInfo(WithPrecision):
    @property
    def dt_type(self) -> str:
        """ discretization method type. """
        return self.extra_attrs["dt_type"]
    @property
    def dt_info(self) -> typing.Any:
        """ discretization information. """
        return self.extra_attrs["dt_info"]
    @dt_info.setter
    def dt_info(self, val):
        assert val is not None
        self.set_extra_attrs(dt_info=val)

#  CustomRulesFuncT = typing.Callable[[WithPrecision], None]
#  """ Rules Definition Function Type
#      @return: how many precisions for current symbol
#               is confirmed.
#  """
#  _CUSTOM_PREC_RULES: typing.Dict[str, CustomRulesFuncT] = {}

#  def custom_prec_rules(*op_names):
#      def _add_rules(f: CustomRulesFuncT):
#          for op in op_names:
#              _CUSTOM_PREC_RULES[op] = f
#          return f
#      return _add_rules

#  def syms_prec(syms: typing.List[WithPrecision], prec: int):
#      for c in syms:
#          c.precision = prec
#  def args_prec(s: WithPrecision, prec: int):
#      return syms_prec(s.args, prec)

#  custom_prec_rules(VAR)(lambda s: 0)
#  custom_prec_rules(ADD, SUB, BIAS_ADD)(lambda s: args_prec(s, 16))
#  custom_prec_rules(CONV2D, DENSE)(lambda s: args_prec(s, 8))
#  custom_prec_rules(MUL, SUM)(lambda s: args_prec(s, 16))

RulesFuncT = typing.Callable[[WithPrecision], None]
_INFER_RULES: typing.Dict[str, RulesFuncT] = {}

def prec_rules(*op_names):
    def _add_rules(f: RulesFuncT):
        for op in op_names:
            _INFER_RULES[op] = f
        return f
    return _add_rules

_infer_mul: RulesFuncT = lambda s: sum([c.precision for c in s.args])
_infer_max: RulesFuncT = lambda s: max([c.precision for c in s.args])

def _infer_index(s: WithPrecision, index: int):
    return s.args[index].precision

prec_rules(TUPLE)(_infer_max)
prec_rules(TUPLE_GET_ITEM)(lambda s: _infer_index(s, s.parsed.index))
@prec_rules(CONV2D, DENSE)
def _infer_nn(s: WithPrecision):
    W = s.args[1]
    add_count = np.product(W.shape[1:])
    add_bits = count_to_bits(add_count)
    return _infer_mul(s) + add_bits
@prec_rules(ADD, SUB)
@prec_rules(BIAS_ADD)
def _infer_add(s: WithPrecision):
    """ op for ADD, SUB should consider scale the same, and then
            to be operated. Here we consider the max precision
            as the max to infer.
    """
    return _infer_max(s) + 1
@prec_rules(CLIP)
@prec_rules(SQUEEZE, RESHAPE)
@prec_rules(RELU, MAX_POOL2D)
def _first_like(s: WithPrecision):
    return _infer_index(s, 0)
@prec_rules(SUM)
def _infer_sum(s: WithPrecision):
    input_len = np.product(s.args[0].shape)
    output_len = np.product(s.shape)
    assert input_len % output_len == 0
    count = int(input_len / output_len)
    sum_bit = count_to_bits(count)
    return _infer_max(s) + sum_bit
prec_rules(MUL)(_infer_mul)
# @prec_rules(CLIP)
# def _infer_clip(s: WithPrecision):
#     a_min = s.attrs["a_min"]
#     a_max = s.attrs["a_max"]
#     absmax = max(math.fabs(a_min), math.fabs(a_max))
#     return number_to_bits(absmax)
@prec_rules(RIGHT_SHIFT)
def _infer_right_shift(s: WithPrecision):
    A, B = s.args[0], s.args[1]
    assert B.is_param()
    b_prec = InferPrecision.bind(B)
    return A.precision - b_prec

@prec_rules(REQUANT, PCLIP, RS_PCLIP)
def _infer_attr_prec(s: WithPrecision):
    return s.parsed.precision

def infer_precision(symbol: WithPrecision) -> int:
    def _infer(sym: Symbol):
        assert sym.op_name in _INFER_RULES, (
                "precision annotator cannot infer op:%s"
                "'s precision."
                ) % sym.op_name
        out = WithPrecision.base(sym)
        out.precision = _INFER_RULES[sym.op_name](sym)
        return out
    out = transform(symbol, _infer)
    return out.precision

@dataclass(repr=False)
class PrecisionAnnotator(WithPrecision, Transformer):
    args: typing.List[PrecisionAnnotator]

    def __call__(self: PrecisionAnnotator):
        if self.op_name in _CUSTOM_PREC_RULES:
            _CUSTOM_PREC_RULES[self.op_name](self)

        if self.is_variable():
            return self
        if self.defined:
            return self

        assert self.op_name in _INFER_RULES, (
                "precision annotator cannot deduct op:%s"
                "'s precision."
                ) % self.op_name
        self.precision = _INFER_RULES[self.op_name](self)
        return self

@dataclass(repr=False)
class InferPrecision(Pass):
    """ Infer Precision Pass

        This inference should be consistent with cvm-runtime.
    """
    args: typing.List[WithPrecision]

    @property
    def arg_precisions(self):
        for a in self.args:
            a.validate_precision()
        return [a.precision for a in self.args]

    def _infer_index(self, index):
        return self.arg_precisions[index]

    def _infer_max(self):
        return max(self.arg_precisions)

    def _infer_mul(self):
        return sum(self.arg_precisions)

    def _first_like(self):
        return self._infer_index(0)

    def _infer_add(self):
        return self._infer_max() + 1

    def _infer_nn(self):
        W = self.args[1]
        add_count = np.product(W.shape[1:])
        add_bits = count_to_bits(add_count)
        return self._infer_mul() + add_bits

# default InferPrecision
@InferPrecision.test(VAR)
def _infer_variable(self: InferPrecision):
    if self.is_input():
        return None
    absmax = np.abs(self.numpy()).max()
    return number_to_bits(absmax)

InferPrecision.test(TUPLE)(InferPrecision._infer_max)
@InferPrecision.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item(self: InferPrecision):
    return self._infer_index(self.parsed.index)
InferPrecision.test(CONV2D, DENSE)(InferPrecision._infer_nn)
InferPrecision.test(BIAS_ADD)(InferPrecision._infer_add)
InferPrecision.test(RELU, MAX_POOL2D)(InferPrecision._first_like)
InferPrecision.test(SQUEEZE, RESHAPE)(InferPrecision._first_like)
@InferPrecision.test(SUM)
def _infer_sum_prec(self: InferPrecision):
    input_len = np.product(self.args[0].shape)
    output_len = np.product(self.shape)
    assert input_len % output_len == 0
    count = int(input_len / output_len)
    sum_bit = count_to_bits(count)
    return self._infer_max() + sum_bit
InferPrecision.test(ADD, SUB)(InferPrecision._infer_add)
InferPrecision.test(MUL)(InferPrecision._infer_mul)
@InferPrecision.test(CLIP)
def _infer_clip(self: InferPrecision):
    a_min = self.attrs["a_min"]
    a_max = self.attrs["a_max"]
    absmax = max(math.fabs(a_min), math.fabs(a_max))
    return number_to_bits(absmax)
@InferPrecision.test(RIGHT_SHIFT)
def _infer_right_shift(self: InferPrecision):
    A, B = self.args[0], self.args[1]
    assert B.is_param()
    b_prec = InferPrecision.bind(B)
    return A.precision - b_prec

def _infer_attr_prec(self: InferPrecision):
    return self.parsed.precision
InferPrecision.test(REQUANT)(_infer_attr_prec)
InferPrecision.test(PCLIP)(_infer_attr_prec)
InferPrecision.test(RS_PCLIP)(_infer_attr_prec)

