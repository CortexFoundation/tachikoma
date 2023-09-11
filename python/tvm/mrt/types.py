import typing

import tvm
import numpy as np

OpOutputT = typing.Union[tvm.nd.NDArray, list]
ParametersT = typing.Dict[str, tvm.nd.NDArray]
AttrsT = typing.Dict[str, typing.Any]

ShapeT = typing.List[int]
""" shape type, list of int, such as [1, 3, 34, 34]. """
DTypeT = str

DataLabelT = typing.Tuple[np.ndarray, typing.Any]
""" a (data, label) representation. """

