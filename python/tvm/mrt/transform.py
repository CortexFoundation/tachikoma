from __future__ import annotations

import typing

import tvm
from tvm import relay, ir

from . import transformers
from .api import Trace, VisitorT, TransformerT
from .extool import *

class Quantizer:
    pass

class Transformer:
    """ Type TransformerT for Trace """

    def __init__(self, tfm_name: str):
        self.name = tfm_name
        self.transformer = getattr(transformers, tfm_name)

    def __call__(self,
            expr: ir.expr.RelayExpr, params: Parameters):
        expr_tfm: TransformerT = getattr(self.transformer,
                "{}_{}".format(self.name, op_name(expr)))
        return expr_tfm(expr, params)

class Quantizer(Transformer):
    def expect_max_precision(self, max_prec) -> Quantizer:
        """ Requantization Method  """
        raise NotImplementedError("")

