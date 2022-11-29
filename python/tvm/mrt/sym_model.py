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
from . import runtime

Visitor = typing.Callable[[Symbol, Parameters], None]
Transformer = typing.Callable[[Symbol, Parameters], typing.Optional[Symbol]]

@dataclass
class Model:
    """ Only use visitor model in Model. """

    symbol: Symbol
    params: Parameters

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

    def random_inputs(self) -> Parameters:
        data = {}
        for sym in self.sym_inputs:
            shape = sym.attrs["shape"]
            dtype = sym.attrs["dtype"]
            np_data = np.random.randn(*shape).astype(dtype)
            data[sym.name] = tvm.nd.array(np_data)
        return data

    def run(self,
            data: typing.Optional[tvm.nd.NDArray] = None,
            data_dict: Parameters = {},
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


    # def set_input_shape(self, shape = None, shape_dict = {}):
    #     shape_dict["common_shape"] = shape
    #     def _set_shape(sym: Symbol):
    #         if is_input(sym, self.params):
    #             shape = shape_dict.get(
    #                     sym.name, shape_dict["common_shape"])
    #             if shape is not None:
    #                 sym.attrs["shape"] = shape
    #         return sym

    #     symbol = transform(self.symbol, _set_shape)
    #     return from_expr(symbol2expr(symbol), self.params)

    def print(self):
        simple_raw_print(self.symbol, self.params)

    def visit(self, callback: Visitor):
        def _visitor(sym: Symbol):
            callback(sym, self.params)
        visit(self.symbol, _visitor)

    def transform(self, callback: Transformer) -> Model:
        def _visitor(sym: Symbol):
            return callback(sym, self.params)
        return transform(self.symbol, _visitor)

    def to_expr(self, expr_map={}) -> ir.RelayExpr:
        return symbol2expr(self.symbol, expr_map)

    # def to_func(self) -> relay.function.Function:
    #     expr_map = {}
    #     expr = self.to_expr(expr_map=expr_map)

    #     # collect params Var
    #     params = [expr_map[s] for s in self.sym_inputs]
    #     params.extend([expr_map[s] for s in self.sym_params])
    #     return relay.Function(params, expr)

    def to_mod(self) -> ir.IRModule:
        return tvm.IRModule.from_expr(self.to_expr())

def from_expr(expr: ir.RelayExpr, params: Parameters) -> Model:
    return Model(expr2symbol(expr), params)

def from_func(func: relay.function.Function, params: Parameters) -> Model:
    return from_expr(func.body, params)

def from_mod(mod: tvm.IRModule, params: Parameters) -> Model:
    return from_func(mod["main"], params)

