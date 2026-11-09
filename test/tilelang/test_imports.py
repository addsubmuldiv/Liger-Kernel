"""Dependency and lazy-import checks that do not require an NPU."""

import ast
import os
import subprocess
import sys

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


TRITON_UNAVAILABLE = r"""
import builtins, importlib.abc, importlib.util, sys
original_import = builtins.__import__
original_find = importlib.util.find_spec
class MissingTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("Triton intentionally unavailable")
def blocked_import(name, *args, **kwargs):
    if name == "triton" or name.startswith("triton."):
        raise ModuleNotFoundError("Triton intentionally unavailable")
    return original_import(name, *args, **kwargs)
def blocked_find(name, *args, **kwargs):
    return None if name == "triton" or name.startswith("triton.") else original_find(name, *args, **kwargs)
sys.meta_path.insert(0, MissingTriton())
builtins.__import__ = blocked_import
importlib.util.find_spec = blocked_find
"""


def run_script(code, **extra_env):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("LIGER_KERNEL_IMPL") or key.startswith("LIGER_KERNEL_BACKEND"):
            env.pop(key)
    env.update(extra_env)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_and_functional_import_without_loading_dsls():
    run_script(
        TRITON_UNAVAILABLE
        + """
import sys
import liger_kernel
import liger_kernel.ops
from liger_kernel.functional import cross_entropy
assert "triton" not in sys.modules
assert "tilelang" not in sys.modules
assert "LigerCrossEntropyFunction" in dir(liger_kernel.ops)
assert "liger_kernel.ops.tilelang.ascend.cross_entropy" in (
    __import__("liger_kernel.backends.registry", fromlist=["_DISCOVERY_MAP"])._DISCOVERY_MAP["cross_entropy"]
)
"""
    )


def test_backend_registration_does_not_import_tilelang():
    run_script("""
import sys
import liger_kernel.ops.tilelang.ascend.cross_entropy
assert "tilelang" not in sys.modules
assert "liger_kernel.ops.tilelang.ascend.ops.cross_entropy" not in sys.modules
""")


def test_default_public_function_export_still_uses_triton():
    pytest.importorskip("triton")
    run_script("""
import liger_kernel.utils
liger_kernel.utils.infer_device = lambda: "cpu"
from liger_kernel.ops import LigerCrossEntropyFunction
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction as TritonFunction
assert LigerCrossEntropyFunction is TritonFunction
""")


def test_missing_tilelang_reports_dependency_error():
    run_script(
        """
import sys
import importlib.abc
import importlib.util
class MissingTileLang(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tilelang" or fullname.startswith("tilelang."):
            raise ModuleNotFoundError("tilelang is intentionally unavailable")
sys.meta_path.insert(0, MissingTileLang())
import liger_kernel.utils
liger_kernel.utils.infer_device = lambda: "npu"
try:
    from liger_kernel.ops import LigerCrossEntropyFunction
except ModuleNotFoundError as exc:
    assert "tilelang" in str(exc)
else:
    raise AssertionError("missing TileLang must not fall back to Triton")
""",
        LIGER_KERNEL_IMPL="ascend-tilelang",
    )


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm", "xpu", "npu"])
def test_setup_keeps_dsls_optional_and_extras_complete(platform):
    tree = ast.parse((ROOT / "setup.py").read_text())
    names = {"get_default_dependencies", "get_triton_dependencies", "get_optional_dependencies"}
    module = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]
    )
    namespace = {"get_project_version": lambda: "0.8.4"}
    exec(compile(module, "setup.py", "exec"), namespace)
    defaults = namespace["get_default_dependencies"](platform)
    extras = namespace["get_optional_dependencies"](platform)
    assert all("triton" not in dep and "tilelang" not in dep for dep in defaults)
    assert extras["tilelang"] == ["tilelang>=0.1.15"]
    expected = (
        []
        if platform == "xpu"
        else ["triton-ascend==3.2.2"]
        if platform == "npu"
        else ["triton>=3.0.0"]
        if platform == "rocm"
        else ["triton>=2.3.1"]
    )
    assert extras["triton"] == expected
    for name in ("cutile", "cutile-tileiras", "cutedsl", "lck"):
        assert all(dep in extras[name] for dep in expected)
