"""Independent CPU-reference tests of the public Ascend950 TileLang CE adapter."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend NPU required", allow_module_level=True)
pytest.importorskip("tilelang.ascend.language")

from liger_kernel.ops.tilelang.ascend.ops.cross_entropy import cross_entropy
from liger_kernel.ops.tilelang.ascend.ops.cross_entropy import cross_entropy_kernel

SHAPES = (
    (128, 32000),
    (1024, 32000),
    (8192, 32000),
    (1269, 32000),
    (128, 128256),
    (1024, 128256),
    (128, 152064),
    (1024, 152064),
)
DTYPES = (torch.bfloat16, torch.float32)


def reference(
    input,
    target,
    weight=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
):
    """Independent CPU FP32 formula/autograd; never calls TileLang or Triton."""
    x = input.float()
    if softcap is not None:
        x = softcap * torch.tanh(x / softcap)
    valid = target != ignore_index
    if not bool(valid.any()):
        loss = x.sum(-1) * 0
        z = loss
    else:
        loss = F.cross_entropy(
            x,
            target.long(),
            weight=weight.float() if weight is not None else None,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        z = torch.where(valid, lse_square_scale * torch.logsumexp(x, dim=-1).square(), 0.0)
        if reduction == "mean":
            denom = valid.sum() if weight is None else weight.float()[target[valid].long()].sum()
            loss = loss / denom
            z = z / valid.sum()
    total = loss + z
    # The Liger interface casts row losses to the input dtype before summing.
    value = total.to(input.dtype)
    logging_z = z.to(input.dtype)
    if reduction != "none":
        value, logging_z = value.sum(), logging_z.sum()
    return value, logging_z


def inputs(N, V, dtype):
    generator = torch.Generator().manual_seed(20261014)
    x = torch.randn((N, V), generator=generator, dtype=dtype)
    y = torch.randint(V, (N,), generator=generator)
    y[: int(N * 0.05)] = -100
    return x, y


def check_case(N, V, dtype, **options):
    cpu, labels = inputs(N, V, dtype)
    ignore_index = options.get("ignore_index", -100)
    labels[labels == -100] = ignore_index
    # Leaves on CPU and NPU are independent and contain identical quantized inputs.
    golden_x = cpu.clone().requires_grad_()
    x = cpu.to("npu").requires_grad_()
    target = labels.to("npu")
    weight = options.pop("weight", None)
    ref_loss, ref_z = reference(golden_x, labels, weight=weight, **options)
    loss, z = cross_entropy(
        x, target, weight=None if weight is None else weight.to("npu"), return_z_loss=True, **options
    )
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(loss.cpu(), ref_loss, atol=1e-8, rtol=rtol)
    torch.testing.assert_close(z.cpu(), ref_z, atol=1e-8, rtol=rtol)
    upstream = torch.full_like(ref_loss, 0.7)
    ref_loss.backward(upstream)
    loss.backward(upstream.to("npu"))
    torch.testing.assert_close(x.grad.cpu(), golden_x.grad, atol=1e-8, rtol=rtol)
    torch.testing.assert_close(x.detach().cpu(), cpu, atol=0, rtol=0)
    torch.testing.assert_close(target.cpu(), labels, atol=0, rtol=0)
    assert loss.dtype == x.dtype and not z.requires_grad
    assert torch.count_nonzero(x.grad[target == ignore_index]).item() == 0


@pytest.mark.parametrize("N,V", SHAPES, ids=[f"n{n}_v{v}" for n, v in SHAPES])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_specified_cases(N, V, dtype):
    check_case(N, V, dtype)


FEATURES = (
    pytest.param("sum", -100, 0.0, None, 0.0, False, id="sum"),
    pytest.param("none", -100, 0.1, None, 0.0, False, id="none-smoothing"),
    pytest.param("mean", -100, 1.0, None, 0.0, False, id="smoothing-one"),
    pytest.param("mean", -1, 0.1, 2.0, 0.0, True, id="weighted-softcap-custom-ignore"),
    pytest.param("sum", -100, 0.2, 3.0, 1e-4, True, id="weighted-softcap-z-sum"),
    pytest.param("none", -100, 0.1, None, 1e-4, True, id="weighted-z-none"),
    pytest.param("mean", -100, 0.0, 5.0, 1e-4, False, id="softcap-z-mean"),
    pytest.param("none", -100, 0.0, 2.0, 0.0, False, id="softcap-none"),
)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction,ignore,smoothing,softcap,z,weighted", FEATURES)
def test_feature_combinations(dtype, reduction, ignore, smoothing, softcap, z, weighted):
    weight = torch.linspace(0.2, 2.0, 73) if weighted else None
    check_case(
        23,
        73,
        dtype,
        weight=weight,
        reduction=reduction,
        ignore_index=ignore,
        label_smoothing=smoothing,
        softcap=softcap,
        lse_square_scale=z,
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "N,V",
    ((1, 1), (1, 7), (57, 8191), (57, 8192), (57, 8193), (3, 65)),
    ids=["single-class", "narrow", "tile-minus-one", "tile-exact", "tile-plus-one", "vector-tail"],
)
def test_boundaries(N, V, dtype):
    check_case(N, V, dtype)


def test_reference_exact_uniform():
    x = torch.zeros((2, 4), requires_grad=True)
    target = torch.tensor([0, 3])
    loss, z = reference(x, target)
    torch.testing.assert_close(loss, torch.tensor(math.log(4)), atol=1e-7, rtol=0)
    assert z.item() == 0
    loss.backward()
    expected = torch.full((2, 4), 0.125)
    expected[0, 0] = expected[1, 3] = -0.375
    torch.testing.assert_close(x.grad, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("reduction", ("mean", "sum", "none"))
def test_all_ignored(dtype, reduction):
    x = torch.randn((3, 65), dtype=dtype, device="npu", requires_grad=True)
    target = torch.full((3,), -100, dtype=torch.int64, device="npu")
    loss, z = cross_entropy(x, target, reduction=reduction, return_z_loss=True, lse_square_scale=1e-4)
    assert torch.count_nonzero(loss).item() == torch.count_nonzero(z).item() == 0
    loss.sum().backward()
    assert torch.count_nonzero(x.grad).item() == 0


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_layout_and_protected_storage(dtype):
    cpu, labels = inputs(23, 73, dtype)
    pool = torch.full((23, 146), 71.0, device="npu", dtype=dtype)
    pool[:, ::2].copy_(cpu.to("npu"))
    original = pool.cpu()
    x = pool[:, ::2].detach().requires_grad_()
    target_pool = torch.full((46,), -7, dtype=torch.int64, device="npu")
    target_pool[::2].copy_(labels.to("npu"))
    target_before = target_pool.cpu()
    weight_pool = torch.linspace(0.1, 2.0, 146, device="npu")
    weight_before = weight_pool.cpu()
    weight = weight_pool[::2]
    cx = cpu.clone().requires_grad_()
    expected, _ = reference(cx, labels, weight=weight_before[::2], reduction="none", label_smoothing=0.1)
    got = cross_entropy(x, target_pool[::2], weight=weight, reduction="none", label_smoothing=0.1)
    up = torch.linspace(-1, 2, 23).to(dtype)
    expected.backward(up)
    got.backward(up.to("npu"))
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(got.cpu(), expected, atol=1e-8, rtol=rtol)
    torch.testing.assert_close(x.grad.cpu(), cx.grad, atol=1e-8, rtol=rtol)
    torch.testing.assert_close(pool.cpu(), original, atol=0, rtol=0)
    torch.testing.assert_close(target_pool.cpu(), target_before, atol=0, rtol=0)
    torch.testing.assert_close(weight_pool.cpu(), weight_before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
def test_repeated_backward_and_inference(dtype):
    cpu, target = inputs(23, 73, dtype)
    x = cpu.to("npu").requires_grad_()
    cx = cpu.clone().requires_grad_()
    loss = cross_entropy(x, target.to("npu"))
    golden, _ = reference(cx, target)
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-6
    for scalar in (1.0, 0.7, -0.2, 1.0):
        upstream = torch.tensor(scalar, dtype=dtype)
        expected = torch.autograd.grad(golden, cx, upstream, retain_graph=True)[0]
        got = torch.autograd.grad(loss, x, upstream.to("npu"), retain_graph=True)[0]
        torch.testing.assert_close(got.cpu(), expected, atol=1e-8, rtol=rtol)
    with torch.no_grad():
        output = cross_entropy(x, target.to("npu"))
    assert not output.requires_grad
    torch.testing.assert_close(output.cpu(), golden.detach(), atol=1e-8, rtol=rtol)
    torch.testing.assert_close(x.detach().cpu(), cpu, atol=0, rtol=0)


def test_extreme_logits():
    cpu = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0], [-1000.0, 1000.0, -1000.0, -1000.0]])
    y = torch.tensor([0, 3])
    cx = cpu.clone().requires_grad_()
    x = cpu.to("npu").requires_grad_()
    expected, _ = reference(cx, y)
    actual = cross_entropy(x, y.to("npu"))
    expected.backward()
    actual.backward()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(x.grad.cpu(), cx.grad, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize(
    "bad",
    (
        "rank",
        "empty",
        "dtype",
        "target-shape",
        "target-dtype",
        "target-negative",
        "target-overflow",
        "target-device",
        "weight-shape",
        "weight-device",
        "weight-grad",
        "reduction",
        "smoothing-low",
        "smoothing-high",
        "softcap",
        "z-loss",
        "ignore-range",
    ),
)
def test_invalid_inputs(bad):
    x = torch.zeros((2, 4), dtype=torch.float32, device="npu", requires_grad=True)
    target = torch.zeros((2,), dtype=torch.int64, device="npu")
    options = {}
    exception, match = ValueError, ""
    if bad == "rank":
        x = x.unsqueeze(0)
        match = "positive shape"
    elif bad == "empty":
        x = x[:0]
        match = "positive shape"
    elif bad == "dtype":
        x = x.to(torch.float16)
        exception, match = TypeError, "bfloat16 or float32"
    elif bad == "target-shape":
        target = target[:1]
        match = "integer vector"
    elif bad == "target-dtype":
        target = target.float()
        match = "integer vector"
    elif bad in ("target-negative", "target-overflow"):
        target[0] = -1 if bad == "target-negative" else 4
        match = "out of bounds"
    elif bad == "target-device":
        target = target.cpu()
        match = "same NPU"
    elif bad == "weight-shape":
        options["weight"] = torch.ones(3, device="npu")
        match = "floating vector"
    elif bad == "weight-device":
        options["weight"] = torch.ones(4)
        match = "floating vector"
    elif bad == "weight-grad":
        options["weight"] = torch.ones(4, device="npu", requires_grad=True)
        match = "weight gradients"
    elif bad == "reduction":
        options["reduction"] = "invalid"
        match = "reduction"
    elif bad.startswith("smoothing"):
        options["label_smoothing"] = -0.1 if bad.endswith("low") else 1.1
        match = "label_smoothing"
    elif bad == "softcap":
        options["softcap"] = 0.0
        match = "softcap"
    elif bad == "z-loss":
        options["lse_square_scale"] = float("nan")
        match = "lse_square_scale"
    else:
        options["ignore_index"] = 2**31
        match = "fit int32"
    with pytest.raises(exception, match=match):
        cross_entropy(x, target, **options)


def test_actual_pto_target():
    kernel = cross_entropy_kernel(1, 1, "float32", 1, 8192, False, 0.0, 0.0, 0.0, True)
    assert "pto" in kernel.target.keys


@pytest.mark.parametrize("bad", ("input-type", "target-type", "weight-type", "return-flag"))
def test_argument_types(bad):
    x = torch.zeros((2, 4), device="npu", requires_grad=True)
    y = torch.zeros((2,), dtype=torch.int64, device="npu")
    options = {}
    if bad == "input-type":
        x = [1, 2]
    elif bad == "target-type":
        y = [0, 0]
    elif bad == "weight-type":
        options["weight"] = [1, 1, 1, 1]
    else:
        options["return_z_loss"] = "yes"
    with pytest.raises(TypeError, match="tensor|bool"):
        cross_entropy(x, y, **options)


def test_double_backward_rejected():
    x = torch.randn((2, 4), device="npu", requires_grad=True)
    y = torch.tensor([0, 3], device="npu")
    loss = cross_entropy(x, y)
    gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
    with pytest.raises(RuntimeError, match="once_differentiable|does not require grad|differentiat"):
        torch.autograd.grad(gradient.sum(), x)
