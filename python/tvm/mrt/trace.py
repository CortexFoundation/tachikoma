from __future__ import annotations
import typing

from dataclasses import dataclass, field
from functools import wraps
import numpy as np

import tvm
from tvm import relay, ir
from tvm.contrib import graph_executor as graph

from .symbol import *
from .types import *
from . import topi
from . import runtime

Visitor = typing.Callable[[Symbol, ParametersT], None]
Transformer = typing.Callable[[Symbol, ParametersT], typing.Optional[Symbol]]

@dataclass
class Trace:
    """ Only use visitor mode in Trace. """

    name: str
    """ Trace Name """
    symbol: Symbol
    params: ParametersT

    sym_inputs: typing.List[Symbol] = field(init=False)
    sym_params: typing.List[Symbol] = field(init=False)

    def __post_init__(self):
        self.sym_inputs = []
        self.sym_params = []
        def _init(sym: Symbol):
            if is_input(sym, self.params):
                self.sym_inputs.append(sym)
            elif is_param(sym, self.params):
                sym_shape = list(sym.attrs["shape"])
                param_shape = self.params[sym.name].shape
                assert sym_shape == list(param_shape), (
                    "param:{} shape inconsistent: {} vs. {}"
                ).format(sym.name, sym_shape, param_shape)
                # sym.attrs["shape"] = self.params[sym.name].shape
                self.sym_params.append(sym)
        visit(self.symbol, _init)

    @property
    def input_names(self) -> typing.List[str]:
        return [i.name for i in self.sym_inputs]

    @property
    def input_shapes(self) -> typing.List[ShapeT]:
        return [i.attrs["shape"] for i in self.sym_inputs]

    def random_inputs(self) -> ParametersT:
        data = {}
        for sym in self.sym_inputs:
            shape = sym.attrs["shape"]
            dtype = sym.attrs["dtype"]
            np_data = np.random.randn(*shape).astype(dtype)
            data[sym.name] = tvm.nd.array(np_data)
        return data

    def calibrate(self,
            data: typing.Optional[np.ndarray] = None,
            data_dict: typing.Dict[str, np.ndarray] = {},
        ) -> typing.Dict[str, np.ndarray]:
        calibrate_outputs: typing.Dict[str, np.ndarray] = {
                k: v.numpy() for k, v in self.params.items()}

        # set input data
        for v in self.sym_inputs:
            shape, dtype = v.attrs["shape"], v.attrs["dtype"]
            val = data_dict.get(v.name, data)
            if val is None:
                print("input: {} use random data".format(v.name))
                val = np.random.randn(*shape).astype(dtype)
            calibrate_outputs[v.name] = val

        def _execute(sym: Symbol, data: ParametersT) -> runtime.OutputDataType:
            args = [ a.as_parameter() for a in sym.args]
            sym = sym.clone(args=args)
            expr = symbol2expr(sym)
            result = runtime.infer(expr, data)
            return result

        def _tassert(expect: typing.Any, val: typing.Any):
            if isinstance(expect, ( list, tuple )):
                assert len(expect) == len(val), (
                    "{} vs. {}").format(expect, val)
                for e, o in zip(expect, val):
                    _tassert(e, o)
            elif isinstance(expect, ( int, str )):
                assert expect == val

        def _get_type(out, key):
            if isinstance(out, tvm.runtime.NDArray):
                return getattr(out, key)
            return [ _get_type(o, key) for o in out ]

        def _calibrate(sym: Symbol, params: ParametersT):
            global TUPLE_GET_ITEM_NAME

            if is_variable(sym, params):
                return
            if sym.op_name == TUPLE_GET_ITEM_NAME:
                out = calibrate_outputs[sym.args[0].name][sym.attrs['index']]
            else:
                out = _execute(sym, calibrate_outputs)

            _tassert(sym.attrs["shape"], _get_type(out, "shape"))
            _tassert(sym.attrs["dtype"], _get_type(out, "dtype"))
            calibrate_outputs[sym.name] = out

        self.visit(_calibrate)
        return calibrate_outputs

    def run(self,
            data: typing.Optional[tvm.nd.NDArray] = None,
            data_dict: ParametersT = {},
            device: tvm.runtime.Device = tvm.runtime.cpu(0),
    ) -> typing.List[np.ndarray]:
        params = {k: v for k, v in self.params.items()}
        for sym in self.sym_inputs:
            val = data_dict.get(sym.name, data)
            shape = sym.attrs["shape"]
            dtype = sym.attrs["dtype"]
            assert val is not None
            assert shape == list(val.shape), (
                    "{}: {} vs. {}").format(
                            sym.name, shape, val.shape)
            assert dtype == val.dtype
            params[sym.name] = val

        return runtime.infer(self.to_mod(), params, device=device)

    def random_run(self) -> typing.List[tvm.nd.NDArray]:
        data = {}
        for sym in self.sym_inputs:
            shape = sym.attrs["shape"]
            dtype = sym.attrs["dtype"]
            np_data = np.random.randn(*shape).astype(dtype)
            data[sym.name] = tvm.nd.array(np_data)
        return self.run(data_dict=data)

    def set_input_shape(self,
            shape = None, shape_dict = {}) -> Trace:
       shape_dict["common_shape"] = shape
       def _set_shape(sym: Symbol):
           if is_input(sym, self.params):
               shape = shape_dict.get(
                       sym.name, shape_dict["common_shape"])
               if shape is not None:
                   sym.attrs["shape"] = shape
           return sym

       symbol = transform(self.symbol, _set_shape)
       return Trace.from_expr(symbol2expr(symbol), self.params)

    def print(self):
        simple_raw_print(self.symbol, self.params)

    def visit(self, callback: Visitor):
        def _visitor(sym: Symbol):
            callback(sym, self.params)
        visit(self.symbol, _visitor)

    def transform(self, callback: Transformer) -> Trace:
        new_params = {k: v for k, v in self.params.items()}
        def _tfm(sym: Symbol):
            return callback(sym, new_params)
        return Trace(callback.__name__,
                transform(self.symbol, _tfm), new_params)

    def to_expr(self, expr_map={}) -> ir.RelayExpr:
        return symbol2expr(self.symbol, expr_map)

    @staticmethod
    def from_expr(expr: RelayExpr, params: ParametersT) -> Trace:
        return Trace("init", expr2symbol(expr), params)

