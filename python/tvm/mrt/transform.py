from __future__ import annotations

import typing
from dataclasses import dataclass, fields

import tvm
from tvm import relay, ir

from . import transformers
from .api import Trace, VisitorT, TransformerT
from .extool import *

@dataclass
class Transformer:
    """ Type TransformerT for Trace """

    op: RelayExpr
    # name: str

    op_name: str = fields(init=False)
    args: typing.List[Transformer] = fields(init=False)
    attrs: typing.Dict[str, typing.Any] = fields(init=False)

    # def __init__(self, tfm_name: str):
    #     self.name = tfm_name
    #     self.transformer = getattr(transformers, tfm_name)

    def __post_init__(self):
        self.op_name = op_name(self.op)
        self.args = args(self.op)
        self.attrs = attrs(self.op)

    # def __call__(self, expr: RelayExpr, params: Parameterss):
    #     expr_tfm: TransformerT = getattr(self.transformer,
    #             "{}_{}".format(self.name, op_name(expr)))
    #     return expr_tfm(expr, params)

    # @classmethod
    # def transform(cls, ):
    

class Validator(Transformer):
    pass

class Quantizer(Transformer):
    def expect_max_precision(self, max_prec) -> Quantizer:
        """ Requantization Method  """
        raise NotImplementedError("")

