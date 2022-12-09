from __future__ import annotations

import math
import typing
from dataclasses import dataclass, field

from .opns import *
from .utils import *
from .symbol import *

from . import op
from .transform import Pass, Transformer
from .calibrate import Calibrator, Sampling, SymmetricMinMaxSampling

@dataclass(repr=False)
class WithPrecision(Symbol):
    precision: int | None

    MAX_BIT: typing.ClassVar[int] = 32

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        return super().default_dict(precision=None, **kwargs)

    def __repr__(self, **attrs):
        if self.precision is not None:
            attrs.setdefault("precision", self.precision)
        return super().__repr__(**attrs)

    def validate_precision(self):
        if self.precision is None:
            return
        assert self.precision <= self.MAX_BIT, (
            "precision:{} out of max bit:{} for \n{}"
        ).format(self.precision, self.MAX_BIT, self)
        assert self.precision > 0

    @classmethod
    def update_dict(cls, data: dict, **kwargs) -> dict:
        prec = data["precision"]
        assert prec is None or prec > 0
        return super().update_dict(data, **kwargs)

@dataclass(repr=False)
class Discretor(Sampling, WithPrecision):
    """ Perform discretization on the sampling data
            and precision.
    """
    info: typing.Any | None
    """ discretization information
            need to provide __eq__ function to compare
            in InferDiscretor.
    """
    precision: typing.Any | None
    """ precision information
            need to provide base arthmetic function
            to compare in InferPrecision.
    """

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        return super().default_dict(
                info=None, _checked=False, **kwargs)

    def __repr__(self, **attrs):
        if self.info is not None:
            attrs.setdefault("discrete_info", self.info)
        return super().__repr__(**attrs)

    # ======== Annotate Functions ==========
    def same(self, other: Discretor) -> Discretor:
        """ make current discretization same as other. """
        return self.copy(
                info=other.info,
                precision=other.precision)

    def set_prec(self, prec: typing.Any) -> Discretor:
        return self.copy(info=None, precision=prec)

    # ======== Quantize Functions ==========
    def summary(self) -> str:
        """ return current discrete information. """
        raise NotImplementedError()

    def mapping(self, sym: Transformer) -> Transformer:
        """ discrete parameters. """
        raise NotImplementedError()

    def remapping(self, base: Discretor, sym: Transformer) -> Transformer:
        self.examine()
        if self.info == base.info:
            return sym
        return self._remapping(base, sym)

    def examine(self):
        """ use calibrated data to revise the info
                and precision.
        """
        self.validate_precision()
        self._examine()
        self.validate_precision()

    def _remapping(self, base, sym):
        raise NotImplementedError()

    def _examine(self):
        raise NotImplementedError()

@dataclass(repr=False)
class SymmetricLinearDiscretor(
        Discretor, SymmetricMinMaxSampling):
    info: float | None
    """ symmetric linear scale to precision integer. """
    precision: int | None

    @classmethod
    def update_dict(cls, data_dict, **kwargs) -> dict:
        if "origin" in data_dict:
            assert isinstance(
                data_dict["origin"], SymmetricMinMaxSampling), \
                type(data_dict["origin"])
        assert isinstance(data_dict["data"], float), \
                type(data_dict["data"])
        return super().update_dict(data_dict, **kwargs)

    def summary(self) -> str:
        return "(scale){},(precision){}".format(
                self.info, self.precision)

    def mapping(self, sym: Symbol) -> Quantizer:
        self.examine()
        out: Quantizer = sym.copy(name=N.n())
        out.update_data(sym.numpy() * self.info)

        # params out of precision will be cliped
        #   in cvm-runtime.
        check = self.sampling(out.numpy())
        checked_bit = number_to_bits(check)
        assert checked_bit <= self.precision
        return out

    def _remapping(self,
            base: SymmetricLinearDiscretor,
            sym: Symbol) -> Quantizer:
        rescale = self.info / base.info
        out = op.requant(sym,
                rescale=rescale,
                precision=self.precision)
        return out

    def _examine(self):
        if self.info is not None:
            real_max = self.data * self.info
            self.precision = number_to_bits(real_max)
        elif self.precision is not None:
            prec_max = 2 ** (self.precision - 1) - 1
            self.info = prec_max / self.data
        else:
            assert False


@dataclass(repr=False)
class ArgDiscretor(Pass):
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

ArgDiscretor.test(VAR)(lambda x: [])
ArgDiscretor.test(CONV2D, DENSE)(ArgDiscretor.with_prec, 8)
ArgDiscretor.test(BIAS_ADD)(ArgDiscretor.first_like)
ArgDiscretor.test(MUL)(ArgDiscretor.with_prec, 16)
ArgDiscretor.test(ADD, SUB)(ArgDiscretor.first_like)
ArgDiscretor.test(TUPLE, TUPLE_GET_ITEM)(ArgDiscretor.identity)
ArgDiscretor.test(SUM)(ArgDiscretor.with_prec, 16)
ArgDiscretor.test(RELU, MAX_POOL2D)(ArgDiscretor.identity)
ArgDiscretor.test(SQUEEZE, RESHAPE)(ArgDiscretor.identity)

@dataclass(repr=False)
class Quantizer(Transformer):
    dt: Discretor | None = field(repr=False)
    revised: Quantizer | None = field(repr=False)
    requants: typing.Dict[str, Quantizer] = field(repr=False)

    @classmethod
    def default_dict(cls, **kwargs) -> dict:
        return super().default_dict(
                dt = None, revised = None,
                requants = {}, **kwargs)

    def __repr__(self, **attrs):
        attrs.setdefault("info", self.dt.summary())
        return super().__repr__(**attrs)

    @classmethod
    def update_dict(cls, data_dict, **kwargs) -> dict:
        data_dict.update(kwargs)
        dt = data_dict.get("dt", None)
        if dt is None and "origin" in data_dict:
            dt : Discretor = data_dict["origin"]
            assert isinstance(dt, Discretor)
        return super().update_dict(data_dict, dt=dt)

    def __repr__(self):
        return super().__repr__(
                info=self.dt and self.dt.summary())

    def __call__(self):
        arg_dts: typing.List[Discretor] = ArgDiscretor.bind(self)

        for i in range(len(self.args)):
            arg: Quantizer = self.args[i].revised
            self.args[i] = arg.requantize(arg_dts[i])

        self.revised = self
        if self.is_operator():
            self.dt.info = InferDiscretor.bind(self)
            self.dt.precision = InferPrecision.bind(self)
            self.examine_precision()
        return InferOperator.bind(self)

    def examine_precision(self):
        """ set revised target with discretor examine.
                use explicit clip to annotate precision
                if necessary.
        """
        new_dt: Discretor = self.dt.copy()
        new_dt.examine()
        out = self
        if self.dt.precision > new_dt.precision:
            out = op.pclip(out, precision=new_dt.precision)
            # print("[    PClip]>> {}".format(out))
        self.revised = out.like(self, dt=new_dt)
        # raw_print(self.revised)

    def requantize(self, new_dt: Discretor):
        new_dt.examine()
        key = new_dt.summary()
        if key in self.requants:
            return self.requants[key]

        if self.is_input():
            out = self.copy()
        elif self.is_param():
            out = new_dt.mapping(self)
        else:
            out = new_dt.remapping(self.dt, self)
        out = out.like(self, dt=new_dt)
        self.requants[key] = out
        return out

@dataclass(repr=False)
class InferPrecision(Pass):
    @property
    def arg_precisions(self):
        return [a.dt.precision for a in self.args]

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

InferPrecision.test(VAR)(lambda x: None)
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
# InferPrecision.test(REQUANT)(InferPrecision._first_like)

@dataclass(repr=False)
class InferDiscretor(Pass):
    @property
    def arg_infos(self):
        return [a.dt.info for a in self.args]

# @dataclass(repr=False)
# class SymmetricLinearInferDiscretor(Pass):
    def infer_index(self, index):
        return self.arg_infos[index]

    def first_like(self):
        infos = self.arg_infos
        assert infos.count(infos[0]) == len(infos), infos
        return self.infer_index(0)

    def infer_mul(self):
        """ only SymmetricLinearDiscretor """
        return np.product(self.arg_infos)

InferDiscretor.test(VAR)(lambda x: None)
InferDiscretor.test(TUPLE)(InferDiscretor.first_like)
@InferDiscretor.test(TUPLE_GET_ITEM)
def _infer_tuple_get_item_scale(self: InferDiscretor):
    return self._infer_index(self.parsed.index)
InferDiscretor.test(CONV2D, DENSE)(InferDiscretor.infer_mul)
InferDiscretor.test(BIAS_ADD)(InferDiscretor.first_like)
InferDiscretor.test(RELU, MAX_POOL2D)(InferDiscretor.first_like)
InferDiscretor.test(SUM)(InferDiscretor.first_like)
InferDiscretor.test(SQUEEZE, RESHAPE)(InferDiscretor.first_like)
InferDiscretor.test(ADD, SUB)(InferDiscretor.first_like)
InferDiscretor.test(MUL)(InferDiscretor.infer_mul)

@dataclass(repr=False)
class InferOperator(Pass):
    def identity(self):
        return self

InferOperator.test_all(InferOperator.identity)

def number_to_bits(number: float) -> int:
    """ Return the integer bits to represent number.
        precision bit: 1
        number bits:
            [ 0-0 ] => 0, skip
            [ 1-1 ] => 1, ceil(log2(i+1)) = 1
            [ 2-3 ] => 2, ceil(log2(i+1)) = 2
            [ 4-7 ] => 3, ceil(log2(i+1)) = 3
            ...

        return 1 + ceil(log2(number + 1))

        note: consider the abs round int for number.
    """
    number = math.fabs(number)
    number = math.floor(number + 0.5)
    return 1 + math.ceil(math.log2(number + 1))

def count_to_bits(count: int):
    """
    # get_bit_cnt (mrt) should be consistent with
    # GetReduceSumBit (cvm-runtime)

    """
    prec = 0
    while count != 0:
        prec += 1
        count >>= 1
    return prec
