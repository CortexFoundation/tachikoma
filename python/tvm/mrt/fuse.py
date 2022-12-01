from dataclasses import InitVar

from .symbol import *

@dataclass
class FusionOp(Symbol):

    params: InitVar[ParametersT]

    def __post_init__(self, params):
        self._fuse_batch_norm()


    @filter_operators("nn.batch_norm")
    def _fuse_batch_norm(self):
        X = self.args[0]
        assert X.is_op("nn.conv2d"), str(self)
        print(self)

        return self

