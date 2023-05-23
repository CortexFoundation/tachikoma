from dataclasses import InitVar
from collections import namedtuple

import numpy as np

from . import op
from .opns import *
from .symbol import *
from .attrs import *
from .utils import N
from .transform import Transformer

class FuseDropout(Transformer):
    @filter_operators(DROP_OUT)
    def __call__(self):
        return self.args[0]

class FuseBatchNorm(Transformer):
    @filter_operators(BATCH_NORM)
    def __call__(self):
        X, gamma, beta, mean, var = self.args
        parsed: BatchNormAttrs = self.parsed

        gamma, beta = gamma.numpy(), beta.numpy()
        mean, var = mean.numpy(), var.numpy()

        beta = beta if parsed.center else 0
        gamma = gamma if parsed.scale else 1

        # (X - mean) / sqrt(var + epsilon) * gamma + beta
        gamma = gamma / np.sqrt(var + parsed.epsilon)
        # (X - mean) * gamma + beta
        # X * gamma + (beta - mean * gamma)
        bias: np.ndarray = (beta - mean * gamma)
        # X * gamma + bias

        if X.is_op(CONV2D):
            A, W = X.args
            conv_parsed: Conv2DAttrs = X.parsed

            assert conv_parsed.kernel_layout == "OIHW"
            K = gamma.shape[0]
            assert W.shape[0] == K

            # (A * W) * gamma + bias
            # A * (W * gamma) + bias
            W_data = W.numpy() * gamma.reshape(K, 1, 1, 1)
            W.update_data(W_data)

            B = X.from_np_data(bias)
            out = op.bias_add(X, B, axis=parsed.axis)
        elif X.is_op(DENSE):
            A, W = X.args
            dense_parsed: DenseAttrs = X.parsed

            # (A * W) * gamma + bias
            # A * (W * gamma) + bias
            W_data = W.numpy() * gamma.reshape(K, 1)
            W.update_data(W_data)

            B = X.from_np_data(bias)
            out = op.bias_add(X, B, axis=parsed.axis)
        else:
            reshp = [s if i == parsed.axis else 1 \
                    for i, s in enumerate(X.shape)]
            W = X.from_np_data(gamma.reshape(reshp))
            out = op.mul(X, W)

            B = X.from_np_data(bias)
            out = op.bias_add(out, B, axis=parsed.axis)
        return out.like(self)

class FuseTupleGetItem(Transformer):
    @filter_operators(TUPLE_GET_ITEM)
    def __call__(self):
        X: Symbol = self.args[0]
        assert X.is_op(BATCH_NORM, DROP_OUT), X.name
        assert self.parsed.index == 0
        return X

class FuseAvgPool2D(Transformer):
    @filter_operators(GLOBAL_AVG_POOL2D)
    def __call__(self):
        X = self.args[0]
        parsed: AvgPool2DAttrs = self.parsed

        assert len(X.shape) == 4
        assert all([s == 1 for s in parsed.output_size])
        assert parsed.layout == "NCHW"
        scale = 1 / np.product(X.shape[-2:])
        out = op.sum(X, axis=list(range(4))[-2:],
                keepdims=True, exclude=False)
        scale = self.from_np_data(scale.astype(X.dtype))
        return op.mul(out, scale).like(self)


