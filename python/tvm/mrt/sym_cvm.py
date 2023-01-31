import typing
from .symbol import *

import cvm

def get_cvm_op(op_name: str) -> typing.Type[cvm.symbol.Symbol]:
    op = getattr(cvm.symbol, op_name, None)

    if op is None:
        raise RuntimeError("cvm not register operator: {}".format(op_name))
    return op

def to_cvm(symbol: Symbol) -> dict:
    cvm_data = {}

    cvm_nodes = []
    for sym in sym2list(symbol):
        cvm_nodes.append({
            "op": sym.op_name,
            "name": sym
            })

