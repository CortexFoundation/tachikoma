from __future__ import annotations
import typing

import os
import pickle
import numpy as np
from functools import wraps
from dataclasses import dataclass, field

import tvm
from tvm import relay, ir
from tvm.contrib import graph_executor as graph

from .symbol import *
from .op import *
from .sym_expr import *
from .types import *
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

    _loaded: bool = False
    _model_name: str = "unknown-model"
    sym_inputs: typing.List[Symbol] = field(init=False)
    sym_params: typing.List[Symbol] = field(init=False)

    BASE_DIR: typing.ClassVar[str] = "./data"

    def __post_init__(self):
        self.sym_inputs = []
        self.sym_params = []
        def _init(sym: Symbol):
            if is_input(sym, self.params):
                self.sym_inputs.append(sym)
            elif is_param(sym, self.params):
                pshape = list(self.params[sym.name].shape)
                assert sym.shape == pshape, (
                    "param:{} shape inconsistent: {} vs. {}"
                ).format(sym.name, sym.shape, pshape)
                self.sym_params.append(sym)
        visit(self.symbol, _init)

        self.params = {s.name: self.params[s.name] \
                for s in self.sym_params}

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

    def print(self, view_layers=0, till_layer=None):
        info = { "op": 0, "param": 0 }
        op_names = set()

        def _simple_visit(sym: Symbol, params: ParametersT):
            if view_layers and info["op"] > view_layers-1:
                return
            if info.get("skip", False):
                return
            if till_layer and till_layer in [
                    sym.name, sym.op_nameA ]:
                info["skip"] = True

            if is_param(sym, params):
                info["param"] += np.product(sym.shape)

            info["op"] += is_operator(sym)

            op_names.add(sym.op_name)
            print(sym)
        self.visit(_simple_visit)

        msg = "{f} {s} {f}".format(f="=" * 25, s=self.name)
        print(msg)
        print("Operators: {} | Parameters: {}".format(
            info["op"], int(info["param"])))
        print(", ".join(op_names))
        print("=" * len(msg))

    def print_ops(self, *op_names: str):
        print("=" * 50)
        print("Collect operators in [{}]".format(
            ", ".join(op_names)))

        @filter_operators(*op_names)
        def _cond_print(sym: Symbol, params: ParametersT):
            print(">", sym)
        self.visit(_cond_print)

        print("=" * 50)

    def subgraph(self, inames=[], onames=[]) -> Trace:
        out = []
        def _find(sym: Symbol, params: ParametersT):
            if sym.name in inames:
                return op.variable(sym.name, sym.shape, sym.dtype)
            elif sym.name  in onames:
                out.append(sym)

        tr = self.transform(_find)
        out = out or [ tr.symbol ]
        out = out[0] if len(out) else op.tuple(*out)
        return Trace("subgraph", out, self.params,
                _loaded=self._loaded,
                _model_name=self._model_name)

    def visit(self, callback: Visitor):
        def _visitor(sym: Symbol):
            callback(sym, self.params)

        with N(callback.__name__):
            visit(self.symbol, _visitor)

    def transform(self,
            callback: Transformer,
            tr_name: str = None,
            print_bf: bool = False,
            print_af: bool = False,
        ) -> Trace:
        tr_name = tr_name or callback.__name__
        new_params = {k: v for k, v in self.params.items()}
        def _tfm(sym: Symbol):
            print_bf and print("[{}]<< {}".format(tr_name, sym))
            out = callback(sym, new_params)
            print_af and print("[{}]>> {}".format(tr_name, out))
            return out

        with N(tr_name):
            new_symbol = transform(self.symbol, _tfm)
        # raw_print(new_symbol)
        print("Applied transform: {}".format(tr_name))
        return Trace(tr_name, new_symbol, new_params,
                _loaded=False,
                _model_name=self._model_name)

    def _get_checkpoint_path(self, tr_name):
        base_dir = os.path.join(self.BASE_DIR, self._model_name)
        os.makedirs(base_dir, exist_ok=True)

        tr_name = tr_name or self.name
        return os.path.join(base_dir, tr_name + ".trace")

    def checkpoint_transform(self,
            *callbacks: Transformer,
            tr_name: str = None,
            force = False,
            **kwargs):
        """ Apply transform in current trace for checkpoint.

            If current trace is not loaded, than force to
                apply transformers.
        """
        assert len(callbacks) > 0
        if not self._loaded:
            force = True

        tr_name = tr_name or callbacks[-1].__name__
        tr_path = self._get_checkpoint_path(tr_name)
        if force or not path.exists(tr_path):
            tr = self
            for cb in callbacks:
                tr = tr.transform(cb, **kwargs)
            tr.dump(tr_path)
            return tr
        tr = Trace.load(tr_path)
        print("Loaded checkpoint: {:20} from {}".format(
            tr_name, tr_path))
        return tr

    def checkpoint(self, tr_name: str = None):
        tr_path = self._get_checkpoint_path(tr_name)
        self.dump(tr_path)

    def dump(self, trace_path: str):
        data = dump_json(self.symbol)
        data.update({
            "_trace_name": self.name,
            "_model_name": self._model_name,
            "params": {k: v.numpy() \
                    for k, v in self.params.items()},
        })
        try:
            with open(trace_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            # clean generated empty path
            os.remove(trace_path)
            raise e

    @staticmethod
    def load(trace_path: str) -> Trace:
        with open(trace_path, "rb") as f:
            data = pickle.load(f)

        name = data["_trace_name"]
        model_name = data["_model_name"]
        params = {k: tvm.nd.array(v) \
                for k, v in data["params"].items()}
        symbol = load_json(data, params=params)
        return Trace(name, symbol, params,
                _loaded=True,
                _model_name=model_name)

    def to_expr(self, expr_map={}) -> ir.RelayExpr:
        return symbol2expr(self.symbol, expr_map)

    @staticmethod
    def from_expr(
            expr: RelayExpr,
            params: ParametersT,
            model_name="unknown-model") -> Trace:
        return Trace("init", expr2symbol(expr), params,
                _loaded=True,
                _model_name=model_name)

