
import numpy as np

import tvm
from tvm import relay, ir, runtime

from .extool import *
from .types import Parameters

def random_inputs(expr: ir.expr.RelayExpr,
        params: Parameters = {}) -> Parameters:
    expr = infer_type(expr)

    free_vars = relay.analysis.free_vars(expr)
    input_data = {k: v for k, v in params.items()}
    for v in relay.analysis.free_vars(expr):
        if v.name_hint in params:
            continue

        print(v.name_hint)

        dtype = v.checked_type.dtype
        shape = v.checked_type.concrete_shape
        np_data = np.random.randn(*shape).astype(dtype)
        input_data[v.name_hint] = tvm.nd.array(np_data)
    return input_data

