from __future__ import annotations

import typing

import tvm
from tvm import relay, ir

from . import transformers

class Quantizer:
    pass

ExprType = typing.Union[relay.expr.Var, relay.expr.Call]

class TransformerExpr(ir.RelayExpr):
    def __init__(self, tfm_name: str):
        self.name = tfm_name
        self.transformer = getattr(transformers, tfm_name)

    def __call__(self, expr: ExprType):
        expr_tfm = getattr(self.transformer,
                "{}_{}".format(self.name, expr.op.name))
        return self.transformer(expr)


class MRTExpr(TransformerExpr):
    def __init__(self):
        super().__init__("quantize")
        self.real_precision = -1

    def expect_max_precision(self, max_prec) -> MRTExpr:
        """ Requantization Method  """
        raise NotImplementedError("")

