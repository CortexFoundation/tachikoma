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

# TODO: add op pass register map.

class FuseDropout(Transformer):
    @filter_operators(DROP_OUT)
    def __call__(self):
        return self.args[0]

class FuseConstant(Transformer):
    def __call__(self: Transformer):
        if self.is_operator() and all([c.is_param() for c in self.args]):
            #  print("fuse constant:", self)
            data = inference.run(
                    self, [c.ndarray() for c in self.args])
            return self.as_parameter(data)
        elif self.is_op(ADD, SUB, BIAS_ADD):
            strips = []
            for arg in self.args:
                if arg.is_param() and np.abs(arg.numpy()).max() == 0:
                    strips.append(arg)
            args = [a for a in self.args if a not in strips]
            if len(args) == 1:
                return args[0]
        elif self.is_op(REQUANT):
            if self.parsed.rescale == 1:
                return self.args[0]

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

        B = out.like(self).from_np_data(bias)
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
    def __call__(self):
        out = self._fuse_adaptive_avg_pool2d()
        out = out or self._fuse_avg_pool2d()
        return out

    @filter_operators(AVG_POOL2D)
    def _fuse_avg_pool2d(self):
        X: Transformer = self.args[0]
        parsed: AvgPool2DAttrs = self.parsed
        assert parsed.layout == "NCHW"
        # TODO: ignore for unstrict mode
        assert parsed.count_include_pad == True
        attrs = {
            "kernel_size": parsed.pool_size,
            "strides": parsed.strides,
            "padding": parsed.padding,
            "dilation": parsed.dilation,
            "data_layout": parsed.layout,
            "groups": X.shape[1],
            "channels": X.shape[1],
            }
        W_shape = (X.shape[1], 1, *parsed.pool_size)
        W = X.from_np_data(np.full(
            W_shape, 1 / product(parsed.pool_size)))
        out = op.nn_conv2d(X, W, **attrs)
        return out.like(self)


    @filter_operators(ADAPTIVE_AVG_POOL2D)
    def _fuse_adaptive_avg_pool2d(self):
        X: Transformer = self.args[0]
        parsed: AdaptiveAvgPool2DAttrs = self.parsed
        assert parsed.layout == "NCHW"
        ins = X.shape[2:]
        ous = parsed.output_size or ins
        if not isinstance(ous, (list, tuple)):
            ous = (ous, ous)
        parsed.output_size = ous

        assert len(X.shape) == 4
        if all([s == 1 for s in parsed.output_size]):
            scale = 1 / np.product(X.shape[-2:])
            out = op.sum(X, axis=list(range(4))[-2:],
                    keepdims=True, exclude=False)
            scale = self.from_np_data(scale.astype(X.dtype))
            return op.mul(out, scale).like(self)
        elif ous[0] > ins[0] or ous[1] > ins[1]:
            assert all([s == 1 for s in ins])
            out = op.repeat(X, repeats=ous[0], axis=-2)
            out = op.repeat(out, repeats=ous[1], axis=-1)
            return out.like(self)

        # calculate the attributes refers to:
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work
        strides = [i // o for i, o in zip(ins, ous)]
        kernel = [i-(o-1)*s for i, o, s in zip(ins, ous, strides)]
        attrs = {
            "kernel_size": kernel,
            "strides": strides,
            "padding": (0, 0),
            "dilation": (1, 1),
            "data_layout": parsed.layout,
            "groups": X.shape[1],
            "channels": X.shape[1],
        }
        W_shape = (X.shape[1], 1, *kernel)
        W = X.from_np_data(np.full(W_shape, 1 / product(kernel)))
        out = op.nn_conv2d(X, W, **attrs)
        return out.like(self)

class FuseNaiveSoftmax(Transformer):
    def __call__(self):
        if self.is_op(SOFTMAX, LOG_SOFTMAX):
            return self.args[0]
        assert self.is_variable() or not self.args[0].is_op(SOFTMAX, LOG_SOFTMAX)
        return self

# move to fuse constant
#  class FuseNaiveMathmatic(Transformer):
#      def __call__(self):
#          if self.is_op(BIAS_ADD):
#              X, B = self.args
#              if B.is_param() and np.abs(B.numpy()).max() == 0:
#                  return X




