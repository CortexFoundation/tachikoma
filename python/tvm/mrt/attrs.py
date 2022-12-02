from functools import wraps
from dataclasses import dataclass, fields

import tvm

from .types import *
from .op import *

@dataclass
class _BaseAttrs:
    shape: typing.Union[ShapeT, typing.List[ShapeT]]
    dtype: typing.Union[DTypeT, typing.List[DTypeT]]

    @classmethod
    def parse(cls, attrs: AttrsT):
        fkeys = [f.name for f in fields(cls) ]
        try:
            data = {k: attrs[k] for k in fkeys}
        except Exception as e:
            print("Attr parsed error, expect: {}, get {}.".format(
                fkeys, attrs.keys()))
            raise e
        return cls(**data)

_ALL_ATTRS = {}

def register_attrs(op_name):
    def _wrapper(cls):
        _ALL_ATTRS[op_name] = cls
        return cls
    return _wrapper

def parse_attrs(op_name, attrs) -> _BaseAttrs:
    if op_name in _ALL_ATTRS:
        return _ALL_ATTRS[op_name].parse(attrs)
    return _BaseAttrs.parse(attrs)

def _format_as_tuple(attrs: AttrsT, *keys):
    for k in keys:
        if k not in attrs:
            continue
        val = attrs[k]
        if not isinstance(val,
                (list, tuple, tvm.ir.container.Array)):
            attrs[k] = [ val, val ]
    return attrs

@dataclass
@register_attrs(TUPLE_GET_ITEM_NAME)
class TupleGetItemAttrs(_BaseAttrs):
    index: int

@dataclass
@register_attrs(CONV2D)
class Conv2DAttrs(_BaseAttrs):
    """ Reference to https://tvm.apache.org/docs/reference/api/python/relay/nn.html#tvm.relay.nn.conv2d

    strides (Optional[int, Tuple[int]]) – The strides of convolution.
    padding (Optional[int, Tuple[int]]) – The padding of convolution on both sides of inputs before convolution.
    dilation (Optional[int, Tuple[int]]) – Specifies the dilation rate to be used for dilated convolution.
    groups (Optional[int]) – Number of groups for grouped convolution.
    channels (Optional[int]) – Number of output channels of this convolution.
    kernel_size (Optional[int, Tuple[int]]) – The spatial of the convolution kernel.
    data_layout (Optional[str]) – Layout of the input.
    kernel_layout (Optional[str]) – Layout of the weight.
    out_layout (Optional[str]) – Layout of the output, by default, out_layout is the same as data_layout
    out_dtype (Optional[str]) – Specifies the output data type for mixed precision conv2d.
    """
    strides: typing.Tuple[int, int]
    padding: typing.Tuple[int, int, int, int]
    dilation: typing.Tuple[int, int]
    groups: int
    channels: int
    kernel_size: typing.Tuple[int, int]
    data_layout: str
    kernel_layout: str
    out_layout: str
    out_dtype: str

    @classmethod
    def parse(cls, attrs: AttrsT):
        attrs = _format_as_tuple(attrs,
                "strides", "dilation", "kernel_size")
        padding = attrs["padding"]
        if isinstance(padding, int):
            padding = [ padding ] * 4
        attrs["padding"] = padding
        return super().parse(attrs)

@dataclass
@register_attrs(BATCH_NORM)
class BatchNormAttrs(_BaseAttrs):
    axis: int = 1
    epsilon: float = 1e-5
    center: bool = True
    scale: bool = True
