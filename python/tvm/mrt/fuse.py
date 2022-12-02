from dataclasses import InitVar
from collections import namedtuple

from .symbol import *
from .attrs import BatchNormAttrs
from .transform import Transformer

from .utils import N

@dataclass
class FusionOp(Transformer):

    new: typing.Optional[Symbol] = None

    def __call__(self):
        self._fuse_batch_norm()

        return self.new


    @filter_operators(BATCH_NORM)
    def _fuse_batch_norm(self):
        X, gamma, beta, mean, var = self.args
        assert X.is_op("nn.conv2d"), str(self)

        parsed: BatchNormAttrs = self.parsed

        gamma, beta = gamma.numpy(), beta.numpy()
        mean, var = mean.numpy(), var.numpy()

        beta = beta if parsed.center else 0
        gamma = gamma if parsed.scale else 1

        # (X - mean) / sqrt(var + epsilon) * gamma - beta
        gamma = gamma / np.sqrt(var + parsed.epsilon)
        # (X - mean) * gamma - beta
        # X * gamma - (beta - mean * gamma)
        bias = beta - mean * gamma
        # X * gamma - bias

        if X.is_op(CONV2D):
            conv_parsed: Conv2DAttrs = X.parsed
            assert conv_parsed.kernel_layout == "OIHW"
            K = gamma.shape[0]
            assert W_data.shape[0] == K

            A, W = X.args
            # (A * W) * gamma - bias
            # A * (W * gamma) - bias
            W_data = W.numpy() * gamma.reshape(K, 1, 1, 1)
            W.update_data(W_data)

            B = self.from_np_data(-bias)

            out = X + B


        print(self)

        params, infer_shapes = kwargs["params"], kwargs["infer_shapes"]
        name = op.attr('name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X, X_name = childs[0], childs[0].attr('name')
        gamma = params[childs[1].attr('name')]
        beta = params[childs[2].attr('name')]
        data_mean = params[childs[3].attr('name')]
        data_var = params[childs[4].attr('name')]

        fix_gamma = get_attr(attr, 'fix_gamma', True)
        gamma = 1 if fix_gamma else gamma
        axis = get_attr(attr, 'axis', 1)

        epsilon = float(attr['eps']) if 'eps' in attr else 1e-5
        sc = gamma / nd.sqrt(data_var + epsilon)
        bias = beta - sc * data_mean

        assert axis == 1, "Channel in input must be axis 1"
        cchilds, cattr = sym_iter(X.get_children()), X.list_attr()

        conv_name = N.n(name)
        W_name = cchilds[1].attr('name')
        weight = params[W_name]
        wn = N.n(W_name)
        params[wn] = weight * sc.reshape(*sc.shape, 1, 1, 1)
        W = mx.sym.var(wn, shape=params[W_name].shape)

        B_name = N.n('bias')
        if not get_attr(cattr, 'no_bias', False):
            B_name = cchilds[2].attr('name')
            bias += params[B_name]
        params[B_name] = bias
        B = mx.sym.var(B_name, shape=bias.shape)

        cattr['no_bias'] = False
        op = mx.sym.Convolution(cchilds[0], W,
                                B, **cattr, name=conv_name)
