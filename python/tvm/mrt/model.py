from .symbol import *

Parameters = typing.Dict[str, tvm.nd.NDArray]

@dataclass
class Model:
    symbol: Symbol
    params: Parameters

def is_operator(self, sym: Symbol):
    pass



