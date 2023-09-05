from __future__ import annotations
import typing

import os
import pickle
import numpy as np
from functools import wraps
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import tvm
from tvm import relay, ir
from tvm.contrib import graph_executor as graph

from . import op
from . import runtime
from .stats import *
from .transform import Transformer
from .discrete import Discretor
from .types import *
from .symbol import *
from .sym_expr import *
from .dataset import Dataset

VisitorT = typing.Callable[[Symbol, ParametersT], None]
TransformerT = typing.Callable[[Symbol, ParametersT], typing.Optional[Symbol]]

@dataclass(repr=False)
class ParamSymbol(Transformer):
    use_all: bool       = False
    use_absmax: bool    = False
    use_shape: bool     = False
    use_dtype: bool     = False

    def __repr__(self, **attrs):
        if self.is_param():
            data = self.numpy()
            if self.use_all or self.use_absmax:
                attrs["absmax"] = np.abs(data).max()
            if self.use_all or self.use_shape:
                attrs["tshape"] = self.shape
            if self.use_all or self.use_dtype:
                attrs["tdtype"] = self.dtype
        return super().__repr__(**attrs)

def uniform_input_data(
        sym_inputs: typing.List[Symbol],
        data: typing.Optional[np.ndarray] = None,
        data_dict: ParametersT = {}):
    input_dict = {}
    for sym in sym_inputs:
        val = data_dict.get(sym.name, data)
        assert val is not None
        #  val = self._preprocess_input(sym, val)
        val = tvm.nd.array(val)
        assert sym.shape == list(val.shape), (
                "{}: {} vs. {}").format(
                        sym.name, sym.shape, val.shape)
        assert sym.dtype == val.dtype, (
                "{} vs. {}").format(sym.dtype, val.dtype)
        input_dict[sym.name] = val
    return input_dict

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
    dataset: typing.optional[Dataset] = field(
            init=False, default=None)
    _executor: graph.GraphModule = field(init=False, default=None)

    uniform_func = uniform_input_data
    BASE_DIR: typing.ClassVar[str] = "./data"

    def __post_init__(self):
        self.sym_inputs = []
        self.sym_params = []
        def _init(sym: Symbol):
            if op.is_input(sym, self.params):
                self.sym_inputs.append(sym)
            elif op.is_param(sym, self.params):
                data = self.params[sym.name]
                assert sym.shape == list(data.shape), (
                    "param:{} shape inconsistent: {} vs. {}"
                ).format(sym.name, sym.shape, pshape)
                assert sym.dtype == data.dtype, (
                    "params:{} dtype inconsistent: {} vs. {}"
                ).format(sym.name, sym.dtype, data.dtype)
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

    def bind_dataset(self, dataset: Dataset, uniform_func = None):
        self.uniform_func = uniform_func or self.uniform_func

        dataset.reset()
        data, label = dataset.next()
        # verify and assert the input data
        input_data = self.uniform_func(self.sym_inputs, data)

        dataset.reset()
        self.dataset = dataset

    def as_input_dict(self,
           data: typing.Optional[np.ndarray] = None,
           data_dict: ParametersT = {},
           ) -> ParametersT:
       input_dict = {}
       for sym in self.sym_inputs:
           val = data_dict.get(sym.name, data)
           assert val is not None
           val = tvm.nd.array(val)
           assert sym.shape == list(val.shape), (
                   "{}: {} vs. {}").format(
                           sym.name, sym.shape, val.shape)
           assert sym.dtype == val.dtype, (
               "{} vs. {}").format(sym.dtype, val.dtype)
           input_dict[sym.name] = val
       return input_dict

    def populate(self, **kwargs) -> runtime.ValidateFunctionT:
        if self._executor is None:
            self._executor = runtime.create_executor(
                    self.to_expr(), self.params,
                    **kwargs)

        def _run(data: np.ndarray) -> np.ndarray:
            # data = self.uniform_func(self.sym_inputs, data)
            data = self.as_input_dict(data)
            # for n, val in data.items():
            #     print("input", n, np.abs(val.numpy()).max(),
            #             val.numpy().flatten()[:5])
            res = runtime.run_executor(self._executor, data)
            assert len(res) == 1
            # print("output", self.symbol, np.abs(res).max())
            return res[0]
        _run.__name__ = self.name
        return _run

    def eval(self,
            data: typing.Optional[np.ndarray] = None,
            **kwargs,) -> np.ndarray:
        return self.populate(**kwargs)(data)

    def infer_type(self) -> Trace:
        tr = Trace.from_expr(
                self.to_expr(), self.params,
                tr_name="infer_type",
                model_name=self._model_name)

        for old, new in zip(
                sym2list(self.symbol), sym2list(tr.symbol)):
            new.attrs.update({
                k: v for k, v in old.attrs.items() \
                    if k not in new.attrs})
            new.extra_attrs.update({
                k: v for k, v in old.extra_attrs.items() \
                    if k not in new.extra_attrs})

        old_sym_iter = iter(sym2list(self.symbol))
        def restore_sym_type(sym: Symbol, params: ParametersT):
            old = next(old_sym_iter)
            new = sym.like(old)
            new.attrs.update({
                k: v for k, v in old.attrs.items() \
                    if k not in new.attrs})
            new.extra_attrs.update({
                k: v for k, v in old.extra_attrs.items() \
                    if k not in new.extra_attrs})
            return new
        restore_sym_type.__name__ = "infer_type"
        tr = tr.transform(restore_sym_type)
        return tr

    def set_input_shape(self,
            shape = None, shape_dict = {},
            tr_name: str = "set_input_shape",
            checkpoint: bool = False,
    ) -> Trace:
        tr_path = self._get_checkpoint_path(tr_name)
        if checkpoint and path.exists(tr_path):
            return Trace.load(tr_path)

        shp_dict = {k: v for k, v in shape_dict.items()}
        for sym in self.sym_inputs:
            shp_dict.setdefault(sym.name, shape or sym.shape)
            if list(shp_dict[sym.name]) != sym.shape:
                print("change {}'s shape from {} into {}".format(
                    sym.name, sym.shape, shp_dict[sym.name]))

        def _set_shape(sym: Symbol, params: ParametersT):
            if op.is_input(sym, params):
                sym.shape = shp_dict[sym.name]
            return sym
        tr = self.transform(_set_shape, tr_name=tr_name)
        tr = tr.infer_type()
        tr.name = tr_name
        tr._loaded = False
        if checkpoint:
            tr.dump(tr_path)
            print("Dumped checkpoint: {:20} into {}".format(
                tr_name, tr_path))
        return tr

    def print(self,
            prefix_layers=0,
            suffix_layers=0,
            short: bool = False,
            till_layer=None,
            selects: typing.List[str] =[],
            param_config = {},
    ):
        msg = "{f} {s} View {f}".format(
                f="=" * 25, s=self.name)
        print(msg)

        info = {
                "ops": 0, "params": 0,
                "op_names": set(),
                "layers": 0, "total_layers": 0,
        }
        def _calc(sym: Symbol, params: ParametersT):
            info["total_layers"] += 1
            info["op_names"].add(sym.op_name)

            if op.is_param(sym, params):
                info["params"] += np.product(sym.shape)
            info["ops"] += op.is_operator(sym)

        self.visit(_calc)

        if short:
            prefix_layers = prefix_layers or 5
            suffix_layers = suffix_layers or 5
        prefix_layers = prefix_layers or info["total_layers"]
        suffix_layers = suffix_layers or info["total_layers"]
        suffix_layers = info["total_layers"] - suffix_layers
        till_layer = till_layer or info["total_layers"]

        user_select = bool(selects)
        selects = selects or info["op_names"]
        # print(prefix_layers, suffix_layers, till_layer)
        def _check(sym: Symbol):
            layer = info["layers"]
            if layer >= till_layer:
                return False
            if layer < prefix_layers or layer >= suffix_layers:
                return True
            return False

        def _print(sym: Symbol, params: ParametersT):
            if suffix_layers > prefix_layers and \
                    info["layers"] == suffix_layers:
                print("\t......\n\t{{skip {} layers}}".format(
                    suffix_layers - prefix_layers))

            checked = _check(sym)
            info["layers"] += 1
            selected = sym.name in selects
            selected = selected or (sym.op_name in selects)
            checked = checked and selected

            if op.is_param(sym, params):
                sym = ParamSymbol.from_dict(
                        sym.to_dict(), params=params,
                        **param_config)
            checked and print(sym)

        self.visit(_print)

        print("_" * len(msg))
        user_select and print("Collect operators in [{}]".format(
            ", ".join(selects)))
        print("Layers: {} | Operators: {} | Parameters: {}".format(
            info["total_layers"], info["ops"], int(info["params"])))
        print("Operator Names:", ", ".join(info["op_names"]))
        print("=" * len(msg))

    def log(self, **kwargs):
        fname = self._get_checkpoint_path(self.name) + ".log"
        print("Log Trace {} into {}".format(
            self.name, fname))
        with open(fname, "w") as f:
            with redirect_stdout(f):
                self.print(**kwargs)

    def subgraph(self, inames=[], onames=[]) -> Trace:
        out = op.subgraph(self.symbol, inames, onames)
        return Trace("subgraph",
                out, self.params,
                _loaded=self._loaded,
                _model_name=self._model_name)

    def visit(self, callback: VisitorT):
        def _visitor(sym: Symbol):
            callback(sym, self.params)

        with N(callback.__name__):
            visit(self.symbol, _visitor)

    def transform(self,
            callback: TransformerT,
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

        print("Apply transformer: {}".format(tr_name))
        with N(tr_name):
            new_symbol = transform(self.symbol, _tfm)

        return Trace(tr_name, new_symbol, new_params,
                _loaded=False, _model_name=self._model_name)


    def _get_checkpoint_path(self, tr_name: str = None):
        base_dir = os.path.join(self.BASE_DIR, self._model_name)
        os.makedirs(base_dir, exist_ok=True)

        tr_name = tr_name or self.name
        return os.path.join(base_dir, tr_name + ".trace")

    def checkpoint_transform(self,
            *callbacks: TransformerT,
            tr_name: str = None,
            force = False,
            **kwargs) -> Trace:
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
                tr = tr.transform(cb, tr_name=tr_name, **kwargs)
            tr.dump(tr_path)
            print("Dumped checkpoint: {:20} into {}".format(
                tr.name, tr_path))
            return tr
        tr = Trace.load(tr_path)
        print("Loaded checkpoint: {:20} from {}".format(
            tr.name, tr_path))
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
    def to_mod(self) -> ir.IRModule:
        return ir.IRModule.from_expr(self.to_expr())

    @staticmethod
    def from_expr(
            expr: RelayExpr,
            params: ParametersT,
            tr_name = "from_expr",
            model_name="unknown-model") -> Trace:
        return Trace(tr_name, expr2symbol(expr), params,
                _loaded=True,
                _model_name=model_name)

