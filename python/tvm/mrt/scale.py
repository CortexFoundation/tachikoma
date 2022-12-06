
from .calibrate import Calibrator
from .transform import Transformer

def number_to_bits(number: float):
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


@dataclass
class Scaler(Transformer):
    data: typing.Any
    scale: typing.Any = 1
    precision: Precision = Precision()

    @classmethod
    def base(cls, sym: Calibrator):
        raise NotImplementedError()

@dataclass
class SymmetricMinMaxScaler(Scaler):
    data: float
    """ threshold for calib data. """
    scale: float

    def __post_init__(self):
        Transformer.__post_init__()
        if self.precision > 0:
            prec_max = 2 ** (self.precision - 1) - 1
            self.scale = self.prec_max / self.data
        elif self.scale != 1:
            real_max = self.data * self.scale
            self.precision = number_to_bits(real_max)

    @classmethod
    def base(cls, sym: Calibrator):
        return cls.from_dict(sym.to_dict(),
                data=sym.np_data.abs().max().scalar())

