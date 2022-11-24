
from tvm import relay, ir

def quantize_conv2d(expr: relay.Call, attrs):
    """ quantize by scale. """
    assert isinstance(expr, relay.Call)
    expr.op
    pass

