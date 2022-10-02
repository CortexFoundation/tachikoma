# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""Tachikoma library supported operators.

From TVM's perspective, Tachikoma is an extension of DNNL. The code below
is adapted from that of DNNL:

There are two ways to registering a function for an op to indicate if it is
supported by Tachikoma.
- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:
    .. code-block:: python
      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)
- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to Tachikoma.
"""
import logging

import tvm.ir
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr import const

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr, rewrite, DFPatternCallback
from .register import register_pattern_table

logger = logging.getLogger("Tachikoma")


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by Tachikoma.
    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.
    Returns
    -------
    f : callable
        A function that returns if the operator is supported by Tachikoma.
    """

    @tvm.ir.register_op_attr(op_name, "target.tachikoma")
    def _func_wrapper(expr):
        args = expr.args
        if any([x.checked_type.dtype == "int64" for x in args]):
            logger.info("Tachikoma does not support int64.")
            return False
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv1d")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.conv3d_transpose")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.max_pool3d")
_register_external_op_helper("nn.avg_pool3d")
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("round")
_register_external_op_helper("logsumexp")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("add")
_register_external_op_helper("multiply")


def make_conv_pattern(conv_name, with_bias=True, with_eltwise=None):
    """Create patterns related to conv and deconv.
    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `conv / deconv`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    conv_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op(conv_name)(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    if with_eltwise:
        return is_op(with_eltwise)(conv_out)
    return conv_out


def make_dense_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.dense.
    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    dense_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    if with_eltwise:
        dense_out = is_op(with_eltwise)(dense_out)
    return dense_out


def make_tachikoma_pattern(op, with_bias, with_eltwise):
    """Create tachikoma patterns.
    Parameters
    ----------
    op : str
        The first call node's op name.
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat_name = op.replace("nn", "tachikoma")
    pat_name += "_bias" if with_bias else ""
    pat_name += ("_" + with_eltwise.split(".")[-1]) if with_eltwise else ""
    if "conv" in op:
        tachikoma_pattern = (pat_name, make_conv_pattern(op, with_bias, with_eltwise))
    elif op == "nn.dense":
        tachikoma_pattern = (pat_name, make_dense_pattern(with_bias, with_eltwise))
    else:
        logger.warning(
            "Currently, only conv1d, conv2d, conv2d_transpose, conv3d_transpose and "
            "dense op are supported, but got %s.",
            op,
        )
        tachikoma_pattern = ()
    return tachikoma_pattern

def make_qnn_conv2d_pattern():
    """Make qnn.conv2d based pattern supported by Tachikoma
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    data = wildcard()
    weight = is_constant()
    # bias = is_constant()
    o_scl = is_op("expand_dims")(is_op("divide")(is_constant(), is_constant()))
    dst_zp = is_op("expand_dims")(is_op("divide")(is_constant(), is_constant())) | is_constant()
    act_scl = is_op("expand_dims")(is_op("divide")(is_constant(), is_constant())) | is_constant()
    sum_scl = is_op("expand_dims")(is_op("divide")(is_constant(), is_constant())) | is_constant()
    sum_src = wildcard()

    zero_zp = is_expr(const(0, dtype="int32"))

    pat = is_op("qnn.conv2d")(data, weight, zero_zp, zero_zp, is_constant(), is_constant())
    pat = is_op("cast")(pat)
    # pat = is_op("add")(pat, bias) | pat  # optional bias
    pat = is_op("multiply")(pat, o_scl)
    pat = is_op("clip")(pat)  # TBD, not only clip
    pat = is_op("multiply")(pat, act_scl) | pat  # optional multiply. Ex: act_scl == 1
    pat = is_op("add")(pat, sum_scl * is_op("cast")(sum_src)) | pat  # optional sum
    pat = is_op("add")(pat, dst_zp) | pat  # optional dst_zp, can be dst_zp == 0
    pat = is_op("cast")(pat)

    return "tachikoma.qnn.conv2d", pat



def make_qnn_dense_pattern():
    """Make qnn.dense based pattern supported by Tachikoma
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    o_scl = is_constant()
    dst_zp = is_constant()
    act_scl = is_constant()
    sum_scl = is_constant()
    sum_src = wildcard()

    zero_zp = is_expr(const(0, dtype="int32"))

    pat = is_op("qnn.dense")(data, weight, zero_zp, zero_zp, is_constant(), is_constant())
    pat = is_op("cast")(pat)
    # pat = is_op("add")(pat, bias) | pat  # optional bias
    pat = is_op("multiply")(pat, o_scl)
    pat = is_op("clip")(pat)  # TBD, not only clip
    pat = is_op("multiply")(pat, act_scl) | pat  # optional multiply. ex act_scl == 1
    pat = is_op("add")(pat, sum_scl * is_op("cast")(sum_src)) | pat  # optional sum
    pat = is_op("add")(pat, dst_zp) | pat  # optional dst_zp, can be dst_zp == 0
    pat = is_op("cast")(pat)

    return "tachikoma.qnn.dense", pat


@register_pattern_table("tachikoma")
def pattern_table():
    """Create tachikoma patterns.
    Returns
    -------
    tachikoma_patterns : List[tachikoma_pattern]
        Created patterns.
    """
    elt_list = ["nn.relu", "tanh", "sigmoid", None]
    tachikoma_patterns = []
    tachikoma_patterns.append(make_qnn_conv2d_pattern())
    tachikoma_patterns.append(make_qnn_dense_pattern())

    for with_bias in [True, False]:
        for elt in elt_list:
            if not with_bias and not elt:
                return tachikoma_patterns
            for conv_name in [
                "nn.conv1d",
                "nn.conv2d",
                "nn.conv3d",
                "nn.conv2d_transpose",
                "nn.conv3d_transpose",
            ]:
                tachikoma_patterns.append(make_tachikoma_pattern(conv_name, with_bias, elt))
            tachikoma_patterns.append(make_tachikoma_pattern("nn.dense", with_bias, elt))
    return tachikoma_patterns

class LegalizeQnnOpForTachikoma(DFPatternCallback):
    """Legalize QNN based patterns to match Tachikoma
    original pattern:
      OP = qnn.dense | qnn.conv2d
      %1 = OP<int>(SRC, WGH) - OP<int>(src_zp, WGH)   // qnn.conv2d
      %2 = %1 + orig_bias                             // bias
      %2 = (%1 - rq_in_zp) * rq_in_scl / rq_out_scl + rq_out_zp  // qnn.requantize
      %3 = act(%2)                                               // activation == clip
      %4 = ((%3 - sum_lh_zp) * sum_lh_scl + (SRC2 - sum_rh_zp) * sum_rh_scl)  // qnn.add
           / sum_out_scl + sum_out_zp
    transform to Tachikoma compatible:
      %1 = OP<int>(SRC, WGH)
      %2 = cast(%1, dtype="float")
      %2 = (%1 + bias) * o_scl
      %3 = act(%2) * act_scl
      %4 = %3 + SRC2 * sum_scl
      %5 = %4 + dst_zp
      %6 = cast(%5, dtype="float")
    where:
      o_scl = rq_in_scl / rq_out_scl
      act_scl = sum_lhs_scl / sum_out_scl
      sum_scl = sum_rhs_scl / sum_out_scl
      bias = orig_bias - OP(src_zp, WGH) - rq_in_zp + rq_out_zp * rq_out_scl / rq_in_scl
      dst_zp = sum_out_zp - sum_lhs_zp * sum_lhs_scl / sum_out_scl -
               sum_rhs_zp * sum_rhs_scl / sum_out_scl
    """

    def __init__(self):
        super(LegalizeQnnOpForTachikoma, self).__init__()
        self.src = wildcard()
        self.wgh = wildcard()
        self.bias = wildcard()
        self.sum_src = wildcard()

        self.src_scl = is_constant()
        self.src_zp = is_constant()
        self.wgh_scl = is_constant()
        self.wgh_zp = is_expr(const(0))

        self.rq_in_scl = is_constant()
        self.rq_in_zp = is_constant()
        self.rq_out_scl = is_constant()
        self.rq_out_zp = is_constant()

        self.sum_lhs_scl = is_constant()
        self.sum_lhs_zp = is_constant()
        self.sum_rhs_scl = is_constant()
        self.sum_rhs_zp = is_constant()
        self.sum_out_scl = is_constant()
        self.sum_out_zp = is_constant()

        self.root = (is_op("qnn.conv2d") | is_op("qnn.dense"))(
            self.src, self.wgh, self.src_zp, self.wgh_zp, self.src_scl, self.wgh_scl
        )
        pat = is_op("add")(self.root, self.bias) | self.root  # optional bias
        pat = is_op("qnn.requantize")(
            pat, self.rq_in_scl, self.rq_in_zp, self.rq_out_scl, self.rq_out_zp
        )
        pat = is_op("clip")(pat)
        cast = is_op("cast")(pat)
        pat = is_op("qnn.add")(
            cast,
            self.sum_src,
            self.sum_lhs_scl,
            self.sum_lhs_zp,
            self.sum_rhs_scl,
            self.sum_rhs_zp,
            self.sum_out_scl,
            self.sum_out_zp,
        )
        pat = is_op("clip")(pat)
        self.pattern = pat | cast

    def callback(self, pre, post, node_map):
        print('callback tachikoma QNN')
        root = node_map[self.root][0]
        src = node_map[self.src][0]
        wgh = node_map[self.wgh][0]
        # bias = node_map.get(self.bias, default=[relay.const(0, dtype="int32")])[0]
        src_zp = node_map[self.src_zp][0]
        rq_in_scl = node_map[self.rq_in_scl][0]
        rq_in_zp = node_map[self.rq_in_zp][0]
        rq_out_scl = node_map[self.rq_out_scl][0]
        rq_out_zp = node_map[self.rq_out_zp][0]

        final_dtype = node_map[self.pattern][0].checked_type.dtype

        if root.op == relay.op.get("qnn.conv2d"):
            dst_layout = root.attrs.out_layout
            dst_layout = root.attrs.data_layout if dst_layout == "" else dst_layout
            wgh_layout = root.attrs.kernel_layout
        else:
            # qnn.dense has no layout attributes. Assume that is plain
            dst_layout = "NC"
            wgh_layout = "OI"

        # TODO(@apeskov): dst_layout may be blocked
        bias_rank = len(dst_layout) - dst_layout.index("C")

        sum_src = node_map[self.sum_src][0] if self.sum_src in node_map else None
        # Default values if qnn.sum is not present
        sum_lhs_scl = node_map[self.sum_lhs_scl][0] if sum_src else relay.const(1, dtype="float32")
        sum_lhs_zp = node_map[self.sum_lhs_zp][0] if sum_src else relay.const(0, dtype="int32")
        sum_rhs_scl = node_map[self.sum_rhs_scl][0] if sum_src else relay.const(0, dtype="float32")
        sum_rhs_zp = node_map[self.sum_rhs_zp][0] if sum_src else relay.const(0, dtype="int32")
        sum_out_scl = node_map[self.sum_out_scl][0] if sum_src else relay.const(1, dtype="float32")
        sum_out_zp = node_map[self.sum_out_zp][0] if sum_src else relay.const(0, dtype="int32")

        def cast_fp(op):
            return relay.op.cast(op, dtype="float32")

        # recalculate some factors
        o_scl = relay.expand_dims(rq_in_scl / rq_out_scl, axis=1, num_newaxis=2)
        act_scl = relay.expand_dims(sum_lhs_scl / sum_out_scl, axis=1, num_newaxis=2) if sum_src else relay.const(1, dtype="float32")
        sum_scl = relay.expand_dims(sum_rhs_scl / sum_out_scl, axis=1, num_newaxis=2) if sum_src else relay.const(0, dtype="float32")
        dst_zp = relay.expand_dims(
            cast_fp(sum_out_zp)
            - cast_fp(sum_lhs_zp) * sum_lhs_scl / sum_out_scl
            - cast_fp(sum_rhs_zp) * sum_rhs_scl / sum_out_scl
            , axis=1, num_newaxis=2) if sum_src else (
            cast_fp(sum_out_zp)
            - cast_fp(sum_lhs_zp) * sum_lhs_scl / sum_out_scl
            - cast_fp(sum_rhs_zp) * sum_rhs_scl / sum_out_scl)
        """
        bias = self.squeeze_bias(bias, dst_layout)
        bias = (
            cast_fp(bias)
            - cast_fp(self.fake_op(src_zp, wgh, wgh_layout))
            - cast_fp(rq_in_zp)
            + cast_fp(rq_out_zp) * rq_out_scl / rq_in_scl
        )
        bias = self.broadcast_to_rank(bias, bias_rank)
        """

        zero_zp = relay.const(0, dtype="int32")
        one_scl = relay.const(1.0, dtype="float32")

        print('1')
        # construct new graph with proper post op ordering
        gr = tvm.relay.Call(
            root.op,
            [src, wgh, zero_zp, zero_zp, one_scl, one_scl],
            root.attrs,
            root.type_args,
            root.span,
        )
        gr = relay.op.cast(gr, dtype="float32")
        # gr = gr + bias
        gr = gr * o_scl
        #print('2')
        gr = relay.op.clip(gr, 0, 255) * act_scl
        gr = gr + sum_scl * cast_fp(sum_src) if sum_src else gr
        gr = gr + dst_zp
        gr = relay.op.cast(gr, dtype=final_dtype)
        print(gr)
        return gr

    @staticmethod
    def fake_op(zp, wgh, layout):
        """Fake operator implementation for zp broadcast input"""
        # Conv:  reduce kernel {OC, IC, KH, KW} -> {OC} in case of group that is still correct
        # Dense: reduce kernel {OC, IC} -> {OC}
        wgh_int = relay.op.cast(wgh, dtype="int32")
        reduced_kernel = relay.op.sum(
            wgh_int, axis=[layout.index("O")], keepdims=False, exclude=True
        )
        return zp * reduced_kernel

    @staticmethod
    def squeeze_bias(bias, layout):
        shape = transform.InferTypeLocal(bias).concrete_shape
        c_position = layout.index("C") - len(layout) + len(shape)
        squeeze_idxs = [i for i in range(len(shape)) if i != c_position]
        return relay.op.squeeze(bias, squeeze_idxs)

    @staticmethod
    def broadcast_to_rank(op, rank):
        """Scalar or 1D tensor are supported"""
        shape = transform.InferTypeLocal(op).concrete_shape
        if len(shape) == 0:
            return op
        if len(shape) == 1:
            return relay.op.expand_dims(op, 1, rank - 1)
        raise ValueError("Unexpected bias rank to broadcast. Only 0 and 1 are supported.")

def legalize_qnn_for_tachikoma(mod):
    """Transform qnn primitives to Tachikoma compatible form. Eliminate source zero point and apply
    strict sequence of post ops."""
    mod["main"] = rewrite(LegalizeQnnOpForTachikoma(), mod["main"])

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            # transform.SimplifyInference(),  # TODO: this pass decompose nn.layer_norm
            # transform.FoldScaleAxis(),  # TODO: fail inside TVM in case of grouped convolutions.
            transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod

def partition_for_tachikoma(mod, params=None):
    """Partition the graph greedily offloading supported operators to Tachikoma.
    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.CanonicalizeOps(),
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
            transform.SimplifyExpr(),
            transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    mod["main"] = rewrite(LegalizeQnnOpForTachikoma(), mod["main"])
    #mod = relay.qnn.transform.CanonicalizeOps()(mod)

    seq_byoc = tvm.transform.Sequential(
        [
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("tachikoma"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq_byoc(mod)

    return mod
