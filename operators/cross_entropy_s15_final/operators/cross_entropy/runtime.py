"""PyTorch host/autograd adapter for the independent TileLang CrossEntropy kernels."""

import math

import torch
import torch_npu  # noqa: F401

from torch.autograd.function import once_differentiable

from operators.cross_entropy.cross_entropy import cross_entropy_kernel
from operators.cross_entropy.cross_entropy import scale_gradient_kernel
from operators.cross_entropy.cross_entropy import simd_deferred_gradient_kernel

DTYPES = (torch.bfloat16, torch.float32)


def _validate(input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap):
    if not isinstance(input, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("input and target must be tensors")
    if input.ndim != 2 or min(input.shape) <= 0:
        raise ValueError("input must have positive shape [N,V]")
    if input.dtype not in DTYPES:
        raise TypeError("input must be bfloat16 or float32")
    if input.device.type != "npu" or target.device != input.device:
        raise ValueError("input and target must be on the same NPU")
    if target.shape != (input.shape[0],) or target.dtype not in (torch.int32, torch.int64):
        raise ValueError("target must be an integer vector of length N")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be none, sum or mean")
    if not isinstance(ignore_index, int) or not -(2**31) <= ignore_index < 2**31:
        raise ValueError("ignore_index must fit int32")
    if not math.isfinite(float(smoothing)) or not 0 <= smoothing <= 1:
        raise ValueError("label_smoothing must lie in [0,1]")
    if not math.isfinite(float(z_scale)) or z_scale < 0:
        raise ValueError("lse_square_scale must be finite and nonnegative")
    if softcap is not None and (not math.isfinite(float(softcap)) or softcap <= 0):
        raise ValueError("softcap must be positive and finite")
    if weight is not None:
        if not isinstance(weight, torch.Tensor):
            raise TypeError("weight must be a tensor or None")
        if weight.shape != (input.shape[1],) or weight.device != input.device or not weight.is_floating_point():
            raise ValueError("weight must be a floating vector of length V on the same NPU")
        if weight.requires_grad:
            raise ValueError("weight gradients are unsupported")


def _forward(input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap, need_grad, defer_grad=False):
    _validate(input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap)
    N, V = input.shape
    valid = target != ignore_index
    invalid = (((target < 0) & valid) | (target >= V)).any()
    # Fetch both scalar reductions together. Separate bool()/item() calls
    # synchronize the host once per result on NPU; this keeps the same checks
    # while reducing the target-validation path to one device-to-host sync.
    invalid_flag, count = torch.stack((invalid.to(torch.int64), valid.sum())).cpu().tolist()
    if invalid_flag:
        raise ValueError("target is out of bounds")
    if weight is None:
        weights = torch.empty((V,), dtype=torch.float32, device=input.device)
        weight_sum = float(V)
        denominator = float(count)
    else:
        weights = weight.float().contiguous()
        weight_sum = weights.sum().item()
        denominator = weights[target[valid].long()].sum().item()
    if reduction != "mean":
        denominator = 1.0
    normalizer = max(1, count) if reduction == "mean" else 1
    # All-ignored rows are zero by contract. Keep the ignored branch free of 0/0.
    if count == 0:
        denominator = 1.0
    x, labels = input.contiguous(), target.to(torch.int32).contiguous()
    simple_path = weight is None and smoothing == 0 and softcap is None and z_scale == 0
    stats = torch.empty((N, 8), device=input.device, dtype=torch.float32) if simple_path else None
    loss = torch.empty((N,), device=input.device, dtype=input.dtype)
    zloss = torch.empty_like(loss)
    # The forward-only specialization never accesses DX; alias X to satisfy its static ABI.
    dx = torch.empty_like(x) if need_grad and not defer_grad else x
    with torch.npu.device(input.device):
        cores = min(N, torch.npu.get_device_properties(input.device).vector_core_num)
        kernel = cross_entropy_kernel(
            N,
            V,
            str(input.dtype).removeprefix("torch."),
            cores,
            8192,
            weight is not None,
            float(smoothing),
            float(softcap or 0),
            float(z_scale),
            need_grad and not defer_grad,
            pipeline_stages=2,
            pipeline_storage="manual",
            save_stats=defer_grad,
        )
        args = (
            x,
            labels,
            weights,
            loss,
            zloss,
            dx,
            ignore_index,
            float(denominator),
            float(normalizer),
            float(weight_sum),
        )
        if simple_path:
            kernel(*args, stats)
        else:
            kernel(*args)
    if reduction != "none":
        loss, zloss = loss.sum(), zloss.sum()
    if defer_grad:
        return loss, zloss, x, labels, stats, float(denominator)
    return loss, zloss, dx


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap):
        need_grad = input.requires_grad
        defer_grad = (
            need_grad
            and weight is None
            and float(z_scale) == 0
            and float(smoothing) == 0
            and softcap is None
            and reduction == "mean"
        )
        if defer_grad:
            loss, zloss, x, labels, stats, denominator = _forward(
                input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap, False, defer_grad=True
            )
            ctx.defer_grad = True
            ctx.ignore_index = ignore_index
            ctx.denominator = denominator
            ctx.save_for_backward(x, labels, stats)
        else:
            loss, zloss, dx = _forward(
                input, target, weight, ignore_index, z_scale, smoothing, reduction, softcap, need_grad
            )
            ctx.defer_grad = False
        if need_grad and not defer_grad:
            ctx.save_for_backward(dx)
        ctx.mark_non_differentiable(zloss)
        return loss, zloss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _grad_z):
        if ctx.defer_grad:
            x, target, stats = ctx.saved_tensors
            N, V = x.shape
            result = torch.empty_like(x)
            upstream = grad_output.contiguous().reshape(-1)
            with torch.npu.device(x.device):
                cores = min(N, torch.npu.get_device_properties(x.device).vector_core_num)
                kernel = simd_deferred_gradient_kernel(
                    N,
                    V,
                    str(x.dtype).removeprefix("torch."),
                    cores,
                    tile=8192,
                    stages=2,
                    storage="manual",
                )
                kernel(x, target, stats, upstream, result, ctx.ignore_index, float(ctx.denominator))
        else:
            (dx,) = ctx.saved_tensors
            if grad_output.ndim == 0 and torch.equal(grad_output, torch.ones((), dtype=dx.dtype, device=dx.device)):
                result = dx
            else:
                N, V = dx.shape
                result = torch.empty_like(dx)
                with torch.npu.device(dx.device):
                    cores = min(N, torch.npu.get_device_properties(dx.device).vector_core_num)
                    kernel = scale_gradient_kernel(
                        N,
                        V,
                        str(dx.dtype).removeprefix("torch."),
                        cores,
                        grad_output.ndim > 0,
                        pipeline_stages=2,
                        pipeline_storage="manual",
                    )
                    kernel(dx, grad_output.contiguous().reshape(-1), result)
        return result, None, None, None, None, None, None, None


def cross_entropy(
    input,
    target,
    weight=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0,
    reduction="mean",
    softcap=None,
    return_z_loss=False,
):
    """Non-mutating CE; an optional detached z-loss is for logging only.

    Input is [N,V], BF16/FP32. All-ignored inputs yield zero loss and gradient.
    Class weights and scalar attributes are constants; double backward and
    the optional Liger accuracy/predicted-token metrics are outside this API.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    if not isinstance(return_z_loss, bool):
        raise TypeError("return_z_loss must be bool")
    if torch.is_grad_enabled() and input.requires_grad:
        loss, zloss = _CrossEntropy.apply(
            input, target, weight, ignore_index, lse_square_scale, label_smoothing, reduction, softcap
        )
    else:
        loss, zloss, _ = _forward(
            input, target, weight, ignore_index, lse_square_scale, label_smoothing, reduction, softcap, False
        )
    return (loss, zloss) if return_z_loss else loss
