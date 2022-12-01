
from .symbol import *

@dataclass
class FusionOp(Symbol):
    def transform(self):
        self = self._fuse_batch_norm()
        return self

    @filter_operators("nn.batch_norm")
    def _fuse_batch_norm(self):
        X = self.args[0]
        assert X.is_op("nn.conv2d"), str(self)
        print(self)

        return self

