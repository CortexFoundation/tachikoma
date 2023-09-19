import typing
from dataclasses import dataclass

from .opns import *
from .transform import Pass, Transformer
from .precision import WithPrecision, QuantizedInfo
from .discrete import Discretor

__ALL__ = [ "ArgAnnotator" ]

@dataclass(repr=False)
class ArgAnnotator(Pass):
    """ Argument Annotator

        Use the pre-defined rules to annotate argument
            discrete information, including precision.

        User can override the rules with Pass `replace`
            helper function.
    """
    args: typing.List[Discretor]

    def with_prec(self, prec: int):
        return [ a.set_prec(prec) for a in self.args ]
        # return [ dt.set_prec(prec) for dt in self.arg_dts ]

    def identity(self):
        return [ a for a in self.args ]
        # return [ dt for dt in self.arg_dts ]

    def first_like(self):
        fdt = self.args[0]
        # fdt = self.arg_dts[0]
        # the first dt should be defined and examined.
        fdt.examine()
        return [ dt.same(fdt) for dt in self.args ]

ArgAnnotator.test(VAR)(lambda x: [])
ArgAnnotator.test(CONV2D, DENSE)(ArgAnnotator.with_prec, 8)
# ArgAnnotator.test(LOG_SOFTMAX)(ArgAnnotator.with_prec, 8)
ArgAnnotator.test(BIAS_ADD)(ArgAnnotator.first_like)
ArgAnnotator.test(MUL)(ArgAnnotator.with_prec, 16)
ArgAnnotator.test(ADD, SUB)(ArgAnnotator.first_like)
ArgAnnotator.test(TUPLE, TUPLE_GET_ITEM)(ArgAnnotator.identity)
ArgAnnotator.test(SUM)(ArgAnnotator.with_prec, 16)
ArgAnnotator.test(RELU, MAX_POOL2D)(ArgAnnotator.identity)
ArgAnnotator.test(SQUEEZE, RESHAPE)(ArgAnnotator.identity)

