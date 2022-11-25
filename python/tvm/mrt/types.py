import typing

import tvm
import numpy as np

Parameters = typing.Dict[str, tvm.nd.NDArray]

ShapeT = typing.Union[typing.List[int], typing.Tuple[int]]
""" shape type, tuple of int, such as (1, 3, 34, 34). """

DataLabelT = typing.Tuple[np.ndarray, typing.Any]
""" a (data, label) representation. """

