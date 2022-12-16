import typing

import numpy as np

import tvm
from tvm import relay, ir, runtime
from tvm.contrib import graph_executor
from tvm.ir import RelayExpr

from .types import *
from .dataset import Dataset
from .stats import Statistics
from . import symbol

__all__ = ["infer"]

def create_executor(
        expr: RelayExpr, params: ParametersT,
        device: tvm.runtime.Device = tvm.runtime.cpu(),
        target: tvm.target.Target = tvm.target.arm_cpu(),
        opt_level=0,
) -> graph_executor.GraphModule:
    target = "llvm"
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build_module.build(
                ir.IRModule.from_expr(expr),
                target=target, params=params)

    rt_mod: graph_executor.GraphModule = \
            graph_executor.GraphModule(lib["default"](device))
    return rt_mod

def run_executor(
        rt_mod: graph_executor.GraphModule,
        input_dict: ParametersT,
        ) -> typing.List[np.ndarray]:
    rt_mod.run(**input_dict)
    return [ rt_mod.get_output(i).numpy() \
            for i in range(rt_mod.get_num_outputs())]

OutputDataType = typing.List[np.ndarray]

def infer(expr: RelayExpr, params: ParametersT,
        device: tvm.runtime.Device = tvm.runtime.cpu(),
        target: tvm.target.Target = tvm.target.arm_cpu(),
) -> OutputDataType:
    # target = tvm.target.cuda()
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build_module.build(
    #             ir.IRModule.from_expr(expr),
    #             target=target,
    #             params=params)

    # rt_mod: relay.build_module.GraphExecutor = graph_executor.GraphModule(lib["default"](device))
    # #  rt_mod.set_input("input", data)
    # rt_mod.run()
    # return [rt_mod.get_output(i).numpy() \
    #         for i in range(rt_mod.get_num_outputs())]

    result = tvm.relay.create_executor(
        "graph", mod=ir.IRModule.from_expr(expr),
        device=device, target=target,
    ).evaluate()(**params)
    return result
    # if isinstance(result, tvm.runtime.NDArray):
    #     result = [ result, ]
    # return [ r.numpy() for r in result ]

def as_numpy(res) -> typing.List[tvm.nd.NDArray]:
    if isinstance(res, tvm.nd.NDArray):
        return [ res.numpy(), ]
    else:
        return [ o.numpy() for o in res ]


def validator(expr: RelayExpr, params: ParametersT, name: str,
        device=runtime.cpu(0), ):
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(
                ir.IRModule.from_expr(expr),
                target=target,
                params=params)
    mod: relay.build_module.GraphExecutor = graph_executor.GraphModule(lib["default"](device))

    input_names = []
    for v in relay.analysis.free_vars(expr):
        if v.name_hint not in params:
            input_names.append(v.name_hint)

    assert len(input_names) == 1
    assert mod.get_num_outputs() == 1
    input_name = input_names[0]

    def _run(dl: DataLabelT) -> DataLabelT:
        data, label = dl
        mod.set_input(input_name, dl)
        mod.run()
        return mod.get_output(0).numpy, dl
    _run.__name__ = name
    return _run


ValidateFunctionT = typing.Callable[[np.ndarray], np.ndarray]

def multiple_validate(
        base_func: ValidateFunctionT,
        dataset: Dataset,
        stats_type: typing.Type[Statistics],
        *comp_funcs: typing.List[ValidateFunctionT],
        max_iter_num: typing.Optional[int] = None,
):
    all_funcs = [ base_func, ] + list(comp_funcs)
    all_stats = [stats_type() for _ in all_funcs]

    log_str = "Iteration: {:3d} | "
    for func in all_funcs:
        log_str += func.__name__ + ": {} | "
    for i in range(max_iter_num or 99999999999999):
        dl = dataset.next()
        if dl is None:
            break
        for func, stats in zip(all_funcs, all_stats):
            out = func(dl[0])
            stats.merge((out, dl[1]))
        msg = log_str.format(i, *[s.info() for s in all_stats])
        print(msg)

    print("Multiple Validation Done!")

