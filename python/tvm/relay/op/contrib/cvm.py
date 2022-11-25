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

import logging

import tvm

from ... import dataflow_pattern as dp
from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr, rewrite, DFPatternCallback
from .register import register_pattern_table

def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by CVMRuntime.

    Parameters
    ----------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.cvm")
    def _func_wrapper(expr: tvm.relay.expr.ExprWithOp):
        # DNNL does not support pooling with ceil_mode = True.
        if "pool" in op_name:
            attrs = dict(get_attrs(expr))
            if "ceil_mode" in attrs.keys() and attrs["ceil_mode"]:
                return False
        return supported

    return _func_wrapper

_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.relu")
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("add")
_register_external_op_helper("bias_add")
_register_external_op_helper("multiply")

def make_nn_pattern(op_name, with_bias=False):
    pat = op_name.replace("nn", "cvm")
    data, weight = wildcard(), wildcard()
    out = dp.is_op(op_name)(data, weight)
    if with_bias:
        bias = wildcard()
        bias_out = dp.is_op("add")(out, bias)
        bias_out = dp.is_op("nn.bias_add")(out, bias) | bias_out
        return [(pat, bias_out), (pat, out)]
    return [(pat, out)]

def make_unary_pattern(op_name):
    pat = "cvm." + op_name
    data = wildcard()
    out = dp.is_op(op_name)(data)
    return [(pat, out)]

def make_binary_pattern(op_name):
    pat = "cvm." + op_name
    A, B = wildcard(), wildcard()
    out = dp.is_op(op_name)(A, B)
    return [(pat, out)]


@register_pattern_table("cvm")
def pattern_table():
    cvm_patterns = list()
    for op_name in ["nn.conv2d", "nn.dense"]:
        cvm_patterns.extend(make_nn_pattern(
            op_name, with_bias=True))
    for op_name in ["nn.relu", "nn.max_pool2d"]:
        cvm_patterns.extend(make_nn_pattern(op_name))

    for op_name in ["abs", "clip"]:
        cvm_patterns.extend(make_unary_pattern(op_name))
    for op_name in ["add", "bias_add", "multiply"]:
        cvm_patterns.extend(make_binary_pattern(op_name))
    return cvm_patterns


def get_op_name(expr):
    """Get the operator name from an expression."""
    if isinstance(expr, Op):
        return expr.name
    if isinstance(expr, Call):
        return get_op_name(expr.op)
    if isinstance(expr, TupleGetItem):
        return get_op_name(expr.tuple_value)
    if isinstance(expr, relay.Tuple):
        return get_op_name(expr.fields[0])
    return ""


def get_args(expr):
    """Get the arguments from an expression."""
    if isinstance(expr, Call):
        return expr.args
    if isinstance(expr, TupleGetItem):
        return get_args(expr.tuple_value)
    if isinstance(expr, relay.Tuple):
        return [arg for args in map(get_args, expr.fields) for arg in args]
    return []


def get_attrs(expr):
    """Get the attributes from an expression."""
    if isinstance(expr, Call):
        return expr.attrs
    if isinstance(expr, TupleGetItem):
        return get_attrs(expr.tuple_value)
    return {}
