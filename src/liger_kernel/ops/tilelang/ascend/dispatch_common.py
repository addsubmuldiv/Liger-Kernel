"""Capability and priority for the optional Ascend TileLang implementation."""

import importlib.util

import torch

from liger_kernel.backends import Capability
from liger_kernel.utils import is_npu_available


def _is_available():
    # Do not import TileLang/TVM while discovering the default Triton backend.
    # CANN's TBE and TileLang use different TVM Python APIs in some releases.
    return (
        is_npu_available()
        and torch.npu.get_device_name().startswith("Ascend950")
        and importlib.util.find_spec("tilelang") is not None
    )


ASCEND_TILELANG_CAPABILITY = Capability(predicate=_is_available)
# Preserve Ascend Triton's existing automatic priority; TileLang is opt-in.
ASCEND_TILELANG_RANK = 20
