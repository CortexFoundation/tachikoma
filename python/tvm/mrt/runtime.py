import typing

import numpy as np

import tvm
from tvm import relay, ir, runtime
from tvm.contrib import graph_executor
from tvm.ir import RelayExpr

from .types import *
from .dataset import Dataset
from .stats import Statistics

__all__ = ["infer"]

def create_executor(
        expr: RelayExpr, params: Parameters, device=runtime.cpu(0),
        opt_level=0
) -> relay.build_module.GraphExecutor:
    target = "llvm"
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build_module.build(
                ir.IRModule.from_expr(expr),
                target=target, params=params)

    rt_mod: relay.build_module.GraphExecutor = \
            graph_executor.GraphModule(lib["default"](device))
    return rt_mod


def infer(expr: RelayExpr, params: Parameters,
        device=runtime.cpu(0)) -> typing.List[np.ndarray]:
    #  target = "llvm"
    target = tvm.target.cuda()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(
                ir.IRModule.from_expr(expr),
                target=target,
                params=params)

    rt_mod: relay.build_module.GraphExecutor = graph_executor.GraphModule(lib["default"](device))
    #  rt_mod.set_input("input", data)
    rt_mod.run()
    return [rt_mod.get_output(i).numpy() \
            for i in range(rt_mod.get_num_outputs())]

    result = tvm.relay.create_executor(
        "graph", mod=ir.IRModule.from_expr(expr),
        device=device, target="llvm"
    ).evaluate()(**params)
    return result.numpy()


def validator(expr: RelayExpr, params: Parameters, name: str,
        device=runtime.cpu(0), ):
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(
                ir.IRModule.from_expr(expr),
                target=target,
                params=params)
    mod: relay.build_module.GraphExecutor = graph_executor.GraphModule(lib["default"](device))

    free_vars = relay.analysis.free_vars(expr)
    assert len(free_vars) == 1
    assert len(mod.get_num_outputs() == 1)
    input_name = free_vars[0].name_hint

    def _run(dl: DataLabelT) -> DataLabelT:
        data, label = dl
        mod.set_input(input_name, dl)
        mod.run()
        return mod.get_output(0).numpy, dl
    _run.__name__ = name
    return _run


ValidateFunctionT = typing.Callable[[DataLabelT], DataLabelT]

def multiple_validate(
        base_func: ValidateFunctionT,
        dataset: Dataset, stats_type: typing.Type[Statistics],
        *comp_funcs: typing.List[ValidateFunctionT],
        max_iter_num: typing.Optional[int] = None,
):
    all_funcs = [ base_func, ] + comp_funcs
    all_stats = [stats_type() for _ in all_funcs]

    log_str = "Iteration: {:3d} | "
    for func in all_funcs:
        log_str += func.__name__ + ": {} | "
    for i in range(max_iter_num):
        dl = dataset.next()
        if dl is None:
            break
        for func, stats in zip(all_funcs, all_stats):
            out_dl = func(dl)
            stats.merge(out_dl)
        msg = log_str.format(i, *[s.info() for s in all_stats])
        print(msg)

    print("Multiple Validation Done!")

