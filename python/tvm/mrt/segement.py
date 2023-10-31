import typing

from dataclasses import dataclass, field

from .symbol import *
from .transform import RunOnce
from . import op, opns

_SCALE_CONSTANT_OPS = [
    opns.VAR,
    opns.WHERE, opns.GREATER,
    opns.REPEAT, opns.SQUEEZE,
    opns.FLATTEN, opns.BATCH_FLATTEN,
    opns.RESHAPE, opns.CONCAT,
    opns.SPLIT, opns.TRANSPOSE,
    opns.STRIDED_SLICE,
    opns.TUPLE, opns.TUPLE_GET_ITEM,
    opns.GET_VALID_COUNT,
    opns.NON_MAX_SUPRESSION,
    opns.CLIP, opns.CAST,
        ]

@dataclass(repr=False)
class Spliter(RunOnce):
    head: typing.Optional[dict] = None
    head_params: typing.Optional[typing.Dict[str, OpNumpyT]] = None
    seg_names: typing.List[str] = field(default_factory=list)

    def __call__(self, **kwargs):
        """ Auto split the model. """
        scans = [ self, ]
        scaned = []
        outs = []
        while scans:
            new_scans = []
            for s in scans:
                if s.name in scaned:
                    continue
                scaned.append(s.name)
                if s.is_op(*_SCALE_CONSTANT_OPS):
                    new_scans.extend(s.args)
                else:
                    outs.append(s)
            scans = new_scans

        self.seg_names = [o.name for o in outs]
        print(self.seg_names)

        def _split(sym: Spliter):
            return op.as_variable(sym) \
                    if sym.name in self.seg_names else sym
        head = transform(self, _split)
        self.head = dump_json(head)

        self.head_params = {}
        def _update_params(sym: Symbol):
            if op.is_param(sym, self.params):
                self.head_params[sym.name] = to_numpy(
                        self.params[sym.name])
        visit(head, _update_params)

        return op.Tuple(*outs).like(self)

@dataclass(repr=False)
class Merger(RunOnce):
    def __call__(self, spliter: Spliter, **kw):
        assert self.op_name == opns.TUPLE
        tail_outs = dict(zip(spliter.seg_names, self.args))

        assert spliter.head is not None
        head_params = {k: to_ndarray(v) \
                for k, v in spliter.head_params.items()}
        head_params.update(self.params)
        head = load_json(spliter.head, params=head_params)

        def _merge(sym: Symbol):
            if op.is_input(sym, head_params):
                assert sym.name in tail_outs
                return tail_outs[sym.name]
            return sym
        out = transform(head, _merge)

        return out.like(self)


