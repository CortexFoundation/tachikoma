import typing
from dataclasses import dataclass

from .opns import *
from .transform import Pass
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
    arg_dts: typing.List[Discretor]

    @classmethod
    def from_dict(cls, data_dict, **kwargs):
        data_dict.update(kwargs)
        # non-revised argument discretor.
        arg_dts = [a.dt.copy() for a in data_dict["args"]]
        return super().from_dict(data_dict, arg_dts=arg_dts)

    def with_prec(self, prec: int):
        return [ dt.set_prec(prec) for dt in self.arg_dts ]

    def identity(self):
        return [ dt for dt in self.arg_dts ]

    def first_like(self):
        fdt = self.arg_dts[0]
        # the first dt should be defined and examined.
        fdt.examine()
        return [ dt.same(fdt) for dt in self.arg_dts ]

ArgAnnotator.test(VAR)(lambda x: [])
ArgAnnotator.test(CONV2D, DENSE)(ArgAnnotator.with_prec, 8)
ArgAnnotator.test(BIAS_ADD)(ArgAnnotator.first_like)
ArgAnnotator.test(MUL)(ArgAnnotator.with_prec, 16)
ArgAnnotator.test(ADD, SUB)(ArgAnnotator.first_like)
ArgAnnotator.test(TUPLE, TUPLE_GET_ITEM)(ArgAnnotator.identity)
ArgAnnotator.test(SUM)(ArgAnnotator.with_prec, 16)
ArgAnnotator.test(RELU, MAX_POOL2D)(ArgAnnotator.identity)
ArgAnnotator.test(SQUEEZE, RESHAPE)(ArgAnnotator.identity)
