"""Numerical coverage of the registered functional and Transformer CE entrypoints."""

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend NPU required", allow_module_level=True)
pytest.importorskip("tilelang.ascend.language")

from test.tilelang.test_cross_entropy import reference
from test.tilelang.test_imports import TRITON_UNAVAILABLE
from test.tilelang.test_imports import run_script

from liger_kernel.functional import cross_entropy as functional_ce
from liger_kernel.ops.tilelang.ascend.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.transformers import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy


@pytest.mark.parametrize("entry", ["functional", "module", "transformers_functional", "function_class"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_public_forward_backward(entry, dtype, reduction, monkeypatch):
    # Per-op selection allows unrelated model operators to keep their default backend.
    monkeypatch.setenv("LIGER_KERNEL_IMPL_CROSS_ENTROPY", "ascend-tilelang")
    cpu = torch.randn(23, 73, dtype=dtype).requires_grad_()
    labels = torch.randint(73, (23,))
    labels[0] = -100
    weight = torch.linspace(0.2, 2.0, 73)
    options = dict(reduction=reduction, label_smoothing=0.1, softcap=2.0, lse_square_scale=1e-4)
    expected, expected_z = reference(cpu, labels, weight=weight, **options)
    x = cpu.detach().npu().requires_grad_()
    target = labels.npu()
    if entry == "functional":
        loss, z, accuracy, tokens = functional_ce(
            x, target, weight=weight.npu(), return_z_loss=True, impl="ascend-tilelang", **options
        )
        assert accuracy is None and tokens is None
    elif entry == "module":
        result = LigerCrossEntropyLoss(weight=weight.npu(), return_z_loss=True, **options)(x, target)
        loss, z = result.loss, result.z_loss
    elif entry == "transformers_functional":
        result = liger_cross_entropy(x, target, weight=weight.npu(), return_z_loss=True, **options)
        loss, z = result.loss, result.z_loss
    else:
        loss, z, accuracy, tokens = LigerCrossEntropyFunction.apply(
            x, target, weight.npu(), -100, 1e-4, 0.1, reduction, 2.0, True
        )
        assert accuracy is None and tokens is None
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(loss.cpu(), expected, atol=1e-8, rtol=rtol)
    torch.testing.assert_close(z.cpu(), expected_z, atol=1e-8, rtol=rtol)
    assert not z.requires_grad
    upstream = torch.full_like(expected, 0.7) if reduction != "none" else torch.linspace(-1, 2, 23).to(dtype)
    expected.backward(upstream)
    loss.backward(upstream.npu())
    torch.testing.assert_close(x.grad.cpu(), cpu.grad, atol=1e-8, rtol=rtol)
    torch.testing.assert_close(x.detach().cpu(), cpu.detach(), atol=0, rtol=0)


@pytest.mark.parametrize("flag", ["return_token_accuracy", "return_predicted_tokens"])
def test_statistics_rejected_explicitly(flag):
    x = torch.randn(3, 65, device="npu", requires_grad=True)
    target = torch.zeros(3, device="npu", dtype=torch.long)
    with pytest.raises(NotImplementedError, match="does not support"):
        functional_ce(x, target, impl="ascend-tilelang", **{flag: True})


def test_mode_and_dtype_rejected():
    x = torch.randn(3, 65, device="npu", requires_grad=True)
    target = torch.zeros(3, device="npu", dtype=torch.long)
    with pytest.raises(ValueError, match="mode"):
        functional_ce(x, target, impl="ascend-tilelang", mode="unsupported")
    with pytest.raises(TypeError, match="bfloat16 or float32"):
        functional_ce(x.half(), target, impl="ascend-tilelang")


def test_legacy_export_and_no_triton_import_in_fresh_process():
    run_script(
        TRITON_UNAVAILABLE
        + """
import torch, torch_npu
from liger_kernel.ops import LigerCrossEntropyFunction, LigerRMSNormFunction
from liger_kernel.transformers import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.functional import cross_entropy
assert LigerCrossEntropyFunction.__module__ == "liger_kernel.ops.tilelang.ascend.ops.cross_entropy"
for call in [lambda x,y: LigerCrossEntropyFunction.apply(x,y)[0],
             lambda x,y: LigerCrossEntropyLoss()(x,y),
             lambda x,y: liger_cross_entropy(x,y),
             lambda x,y: cross_entropy(x,y,impl="ascend-tilelang")[0]]:
    x = torch.zeros(2,4,device="npu",requires_grad=True)
    y = torch.tensor([0,3],device="npu")
    loss = call(x,y)
    loss.backward()
    expected = torch.full((2,4), .125)
    expected[0,0] = expected[1,3] = -.375
    torch.testing.assert_close(x.grad.cpu(), expected, atol=0, rtol=0)
try:
    LigerRMSNormFunction.apply(torch.ones(2,4,device="npu"))
except RuntimeError as exc:
    assert "does not provide" in str(exc)
else:
    raise AssertionError("Unsupported TileLang op must not use Triton")
assert not any(k == "triton" or k.startswith("triton.") for k in sys.modules)
""",
        LIGER_KERNEL_IMPL="ascend-tilelang",
        TILELANG_DEFAULT_TARGET="pto",
    )


def test_default_triton_selection_does_not_load_tilelang():
    run_script("""
import sys, torch, torch_npu
from liger_kernel.backends import resolve_impl
from liger_kernel.functional import cross_entropy
assert resolve_impl("cross_entropy",device=torch.device("npu:0")).impl_name == "ascend-triton"
x = torch.randn(3,65,device="npu",requires_grad=True)
y = torch.zeros(3,device="npu",dtype=torch.long)
loss = cross_entropy(x,y)[0]
loss.backward(torch.tensor(.7,device="npu"))
assert "tilelang" not in sys.modules
""")


@pytest.mark.parametrize("order", ["tilelang_first", "triton_first"])
def test_both_backends_can_run_in_one_process(order):
    names = ["ascend-tilelang", "ascend-triton"]
    if order == "triton_first":
        names.reverse()
    run_script(f"""
import os, torch, torch_npu
os.environ.pop("ASCEND_UB_CAPACITY_BITS", None)
from liger_kernel.functional import cross_entropy
from liger_kernel.ops.backends._ascend.ub_manager import get_ub_manager
x_cpu = torch.randn(3,65,requires_grad=True)
y_cpu = torch.tensor([0,2,64])
ref = torch.nn.functional.cross_entropy(x_cpu,y_cpu)
ref.backward(torch.tensor(.7))
for name in {names!r}:
    x = x_cpu.detach().npu().requires_grad_()
    loss = cross_entropy(x,y_cpu.npu(),impl=name)[0]
    loss.backward(torch.tensor(.7,device="npu"))
    torch.testing.assert_close(loss.cpu(),ref.detach(),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(x.grad.cpu(),x_cpu.grad,atol=1e-6,rtol=1e-5)
assert get_ub_manager().ub_capacity_bits > 0
""")


def test_ub_configuration_requires_exact_soc(tmp_path, monkeypatch):
    from liger_kernel.ops.backends._ascend.ub_manager import _ascend950_ub_capacity_from_config

    directory = tmp_path / "data" / "platform_config"
    directory.mkdir(parents=True)
    path = directory / "Ascend950DT_9582.ini"
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    path.write_text("[version]\nSoC_version=Ascend950PR_9579\n[AICoreSpec]\nub_size=253952\n")
    assert _ascend950_ub_capacity_from_config("Ascend950DT_9582") is None
    path.write_text("[version]\nSoC_version=Ascend950DT_9582\n[AICoreSpec]\nub_size=253952\n")
    assert _ascend950_ub_capacity_from_config("Ascend950DT_9582") == 253952 * 8
    assert _ascend950_ub_capacity_from_config("Ascend910B2") is None
