# TileLang backend: Ascend950 CrossEntropy

This backend follows the optional DSL integration in [PR #1572](https://github.com/linkedin/Liger-Kernel/pull/1572), pinned for this implementation to `b8c8858efdecd5350724de12a488c8f85c18c8d5`. This branch supplies **CrossEntropy only**, using the S15 kernel. It does not copy the reference PR's SwiGLU or RMSNorm implementations.

## Installation

Triton and TileLang are separate optional extras. For the usual Triton backend, install `pip install -e ".[triton]"` (or `".[dev,triton]"` for development). Existing cuTile/CuTe DSL extras also include their Triton requirements. The AMD and Modal jobs request the Triton extra explicitly.

For TileLang, configure a matching CANN/PTOAS and PyTorch/torch_npu environment, then install `pip install -e ".[tilelang]"`. An Ascend-capable TileLang build is required: the PyPI version alone does not guarantee that `tilelang.ascend.language` and the PTO backend are available. Validation uses TileLang source commit [`fd2ddda1a7b8dec637c592eb92bbca46030ed585`](https://github.com/tile-ai/tilelang/commit/fd2ddda1a7b8dec637c592eb92bbca46030ed585). If the compatible stack is already installed, use `pip install -e . --no-deps` to retain it. Platform-specific Torch pins remain unchanged.

Select the compiler backend at process launch:

```bash
TILELANG_DEFAULT_TARGET=pto python your_script.py
```

## Select one functional call

```python
import torch
import torch_npu

from liger_kernel.functional import cross_entropy

x = torch.randn(128, 32000, device="npu", dtype=torch.float32, requires_grad=True)
target = torch.randint(32000, (128,), device="npu")
loss, z_loss, token_accuracy, predicted_tokens = cross_entropy(
    x, target, impl="ascend-tilelang"
)
loss.backward()
```

The public functional API returns the usual four-element tuple. Optional outputs are `None` unless requested. Detached z-loss logging is supported; token accuracy and predicted-token outputs currently raise `NotImplementedError`.

For an environment that also has Triton installed, select CE alone while leaving other model operators on their existing backends:

```bash
LIGER_KERNEL_IMPL_CROSS_ENTROPY=ascend-tilelang TILELANG_DEFAULT_TARGET=pto python your_script.py
```

This applies to `liger_kernel.functional.cross_entropy`, `LigerCrossEntropyLoss` and `liger_kernel.transformers.functional.liger_cross_entropy`.

## Public Function and Triton-free imports

Select the implementation **before importing** the public Function or Transformer classes:

```bash
LIGER_KERNEL_IMPL=ascend-tilelang TILELANG_DEFAULT_TARGET=pto python your_script.py
```

```python
from liger_kernel.ops import LigerCrossEntropyFunction
from liger_kernel.transformers import LigerCrossEntropyLoss

loss, z_loss, accuracy, tokens = LigerCrossEntropyFunction.apply(x, target)
# Or:
loss = LigerCrossEntropyLoss()(x, target)
```

Public operators are loaded on demand, so these TileLang paths do not require importing Triton. Selecting TileLang globally does **not** silently run unsupported operators through Triton: other public operator calls raise `ImplNotAvailableError`. Use per-op selection when a model needs other Triton operators. Direct imports from historical modules such as `liger_kernel.ops.cross_entropy` continue to name the original Triton implementation.

## Supported contract

- Ascend950, BF16 or FP32 logits `[N,V]`, positive dimensions, integer targets `[N]` on the same NPU.
- `mean` / `sum` / `none`, ignored labels, constant class weights, label smoothing, softcap and z-loss regularization.
- First-order input gradients, repeated backward, and inference; non-contiguous inputs are materialized as needed.
- Inputs, targets and class weights are not modified. All-ignored inputs return zero loss and gradients, including for `mean`.
- FP16 inputs, higher-order gradients, class-weight gradients, token accuracy and predicted tokens are explicitly unsupported.
- The numerical kernel ASTs are unchanged from the tested S15 candidate. The host adapter exposes Liger's public four-output Function ABI, and an inference path avoids computing unused input gradients.

On the validated CANN stack, Triton and TileLang can also be selected sequentially in one process. Ascend950 UB detection reads the active CANN's full-SoC-matched platform configuration before using the legacy TBE fallback. This avoids importing CANN's old TVM into a process using TileLang's TVM; it does not hardcode a new buffer size, change existing explicit UB overrides, or change other SoCs' query path.

## Validation

```bash
TILELANG_DEFAULT_TARGET=pto python -m pytest test/tilelang/ -q
```

Tests cover the migrated 79-case independent CPU-reference suite, functional/module/Function/Transformer entrypoints, per-op selection, strict unsupported-operation errors, missing optional dependencies, no-Triton imports, unchanged default Triton selection, and both backend invocation orders. Backend-registry tests run separately under `test/backends/`. Optional hardware/dependency skips remain explicit on other machines.

Executed on Ascend950DT_9582 with PyTorch 2.9.0 / torch_npu 2.9.0.post4, CANN 9.2.0 B237 and the TileLang commit above:

- 79 independent numerical cases passed.
- 29 public entrypoint cases and 3 coexistence/configuration cases passed.
- 80 import/packaging/backend cases passed; 16 conditional hardware cases skipped.
- `make checkstyle` passed (503 Python files unchanged).
- The built wheel contains the backend, matches the tested package sources and passes public-module forward/backward on NPU.

These are scoped checks (191 passed, 16 skipped), not the complete repository or model convergence suites.

## Integrated public-path performance

The four default FP32 benchmark shapes are `[N, 128256]`, with `N = 1024, 2048, 4096, 8192`. Both providers' loss and input gradients passed a separate CPU FP64 reference on every shape before timing. Settings are mean reduction, no class weights/smoothing/softcap/z-loss, upstream gradient 0.7.

Timings below are NPU Events around public functional calls on physical device 1: 10 warmups and 30 samples for each provider order, 192 MiB cache clearing before the start event. Input cloning is outside timing. Backward-only timings use a new forward graph outside timing for each sample. Each table latency is the geometric mean of the two order-specific medians.

| N | Triton forward + backward (µs) | TileLang forward + backward (µs) | Speedup |
|---:|---:|---:|---:|
| 1024 | 1056.3 | 752.9 | 1.403x |
| 2048 | 1667.2 | 1249.9 | 1.334x |
| 4096 | 3032.2 | 2173.4 | 1.395x |
| 8192 | 5775.9 | 4062.9 | 1.422x |

Geometric mean speedups (Triton / TileLang): forward **1.542x**, backward **1.222x**, forward + backward **1.388x**, no-grad forward **1.560x**. All four forward + backward shapes improved. These measure complete operator calls, not isolated kernels or model throughput. BF16 correctness was tested; these performance numbers are FP32 only.

See [`tilelang-cross-entropy-validation.json`](tilelang-cross-entropy-validation.json) for source hashes, per-order medians, environment and test records. Raw timing samples, the measurement script and JUnit logs are retained in the recorded local artifact directory. Earlier standalone S15 measurements (1.403x / 1.617x) are historical and do not describe this dispatch path.
