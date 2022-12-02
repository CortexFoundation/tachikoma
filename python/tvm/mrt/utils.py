from __future__ import annotations

import typing
import threading
import os
from os import path

from tvm import relay, ir

from .types import *

ROOT = path.abspath(path.join(__file__, "../../../"))
PY_ROOT = path.join(ROOT, "python")

MRT_MODEL_ROOT = path.expanduser("~/mrt_model")
if not path.exists(MRT_MODEL_ROOT):
    os.makedirs(MRT_MODEL_ROOT)

MRT_DATASET_ROOT = path.expanduser("~/.mxnet/datasets")
if not path.exists(MRT_DATASET_ROOT):
    os.makedirs(MRT_DATASET_ROOT)

def product(shape: ShapeT):
    total = 1
    for s in shape:
        total *= s
    return total

class N:
    def __init__(self, name=""):
        self.counter = 0
        self.scope_name = name
        self.lock = threading.Lock()
        self.last_scope = N.__GLOBAL_INSTANCE__

    def __enter__(self):
        self._set_name_scope(self)
        return self

    def __exit__(self, *args):
        self._set_name_scope(self.last_scope)

    def _alloc_name(self, prefix, suffix):
        with self.lock:
            index = self.counter
            self.counter += 1
        name = "{}{}{}".format(prefix, index, suffix)
        if self.scope_name:
            name = "{}.{}".format(self.scope_name, name)
        return name

    __GLOBAL_INSTANCE__ = None

    @staticmethod
    def _set_name_scope(ins):
        N.__GLOBAL_INSTANCE__ = ins

    @staticmethod
    def n(prefix=None, suffix=None):
        ins = N.__GLOBAL_INSTANCE__
        if ins is None:
            raise RuntimeError("Namescope not specified")
        prefix = "%" if prefix is None else prefix
        suffix = "" if suffix is None else suffix
        return ins._alloc_name(prefix, suffix)

    @staticmethod
    def register_global_scope(name=""):
        N._set_name_scope(N(name))

def extend_fname(prefix, with_ext=False):
    """ Get the precision of the data.

        Parameters
        __________
        prefix : str
            The model path prefix.
        with_ext : bool
            Whether to include ext_file path in return value.

        Returns
        _______
        ret : tuple
            The symbol path, params path; and with_ext is True, also return ext file path.
    """
    ret = ["%s.json"%prefix, "%s.params"%prefix]
    if with_ext:
        ret.append("%s.ext"%prefix)
    return tuple(ret)
