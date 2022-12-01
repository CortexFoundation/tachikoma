import typing

import tvm
import numpy as np

ParametersT = typing.Dict[str, tvm.nd.NDArray]
AttrsT = typing.Dict[str, typing.Any]

ShapeT = typing.Union[typing.List[int], typing.Tuple[int]]
""" shape type, tuple of int, such as (1, 3, 34, 34). """

DataLabelT = typing.Tuple[np.ndarray, typing.Any]
""" a (data, label) representation. """

