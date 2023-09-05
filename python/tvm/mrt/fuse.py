from dataclasses import InitVar
from collections import namedtuple

import numpy as np

from . import op, inference
from .opns import *
from .symbol import *
from .attrs import *
from .utils import N
from .transform import Transformer
from .inference import np_executor, run

class FuseDropout(Transformer):
    @filter_operators(DROP_OUT)
    def __call__(self):
        return self.args[0]

class FuseConstant(Transformer):
    def __call__(self: Transformer):
        if self.is_operator() and all([c.is_param() for c in self.args]):
            #  print("fuse constant:", self)
            data = inference.run(self, [c.numpy() for c in self.args])
            return self.as_parameter(data)

            #  inputs = [c.numpy() for c in self.args]
            #  data = np_executor(self, inputs)
            #  out = self.from_np_data(data)
            #  return out

class FuseBatchNorm(Transformer):
    @filter_operators(BATCH_NORM)
    def __call__(self):
        X, gamma, beta, mean, var = self.args
        parsed: BatchNormAttrs = self.parsed

        gamma, beta = gamma.numpy(), beta.numpy()
        mean, var = mean.numpy(), var.numpy()
        #  print(gamma.shape, beta.shape, mean.shape, var.shape)

        assert parsed.axis == 1
        beta = beta if parsed.center else 0
        gamma = gamma if parsed.scale else 1

        # (X - mean) / sqrt(var + epsilon) * gamma + beta
        gamma = gamma / np.sqrt(var + parsed.epsilon)
        # (X - mean) * gamma + beta
        # X * gamma + (beta - mean * gamma)
        bias: np.ndarray = (beta - mean * gamma)
        #  print(np.abs(gamma).max(), np.abs(bias).max())
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
            W_sym = W.from_np_data(W_data)
            out = op.nn_conv2d(A, W_sym, **X.attrs)
        elif X.is_op(DENSE):
            A, W = X.args
            dense_parsed: DenseAttrs = X.parsed

            # (A * W) * gamma + bias
            # A * (W * gamma) + bias
            W_data = W.numpy() * gamma.reshape(K, 1)
            W_sym = W.from_np_data(W_data)
            out = op.nn_dense(A, W_sym, **X.attrs)
        else:
            reshp = [s if i == parsed.axis else 1 \
                    for i, s in enumerate(X.shape)]
            W = X.from_np_data(gamma.reshape(reshp))
            out = op.mul(X, W)

        B = self.from_np_data(bias)
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
        parsed: GlobalAvgPool2DAttrs = self.parsed

        assert len(X.shape) == 4
        assert all([s == 1 for s in parsed.output_size])
        assert parsed.layout == "NCHW"
        scale = 1 / np.product(X.shape[-2:])
        out = op.sum(X, axis=list(range(4))[-2:],
                keepdims=True, exclude=False)
        scale = self.from_np_data(scale.astype(X.dtype))
        return op.mul(out, scale).like(self)

class FuseNaiveSoftmax(Transformer):
    def __call__(self):
        if self.is_op(SOFTMAX, LOG_SOFTMAX):
            return self.args[0]
        assert self.is_variable() or not self.args[0].is_op(SOFTMAX, LOG_SOFTMAX)
        return self

class FuseNaiveMathmatic(Transformer):
    def __call__(self):
        if self.is_op(BIAS_ADD):
            X, B = self.args
            if B.is_param() and np.abs(B.numpy()).max() == 0:
                return X




