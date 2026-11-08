# Standalone Ascend950 CrossEntropy (S15)

This directory provides the S15 TileLang/PTO CrossEntropy implementation and an independent CPU-reference test suite for Ascend950. It is a standalone operator snapshot: it does not change Liger's public imports, backend selection, Transformers integrations, or package dependencies. The `operators/` snapshot is not included in the installed `liger_kernel` Python package.

## Requirements and execution

Use a configured Ascend NPU environment with matching PyTorch / torch_npu, CANN and PTOAS installations. The tested TileLang compiler is commit [`fd2ddda1a7b8dec637c592eb92bbca46030ed585`](https://github.com/tile-ai/tilelang/commit/fd2ddda1a7b8dec637c592eb92bbca46030ed585), built with Ascend support. Select the PTO backend at process launch; the source does not hardcode a backend.

From the repository root, after activating that environment and sourcing its CANN configuration:

```bash
cd operators/cross_entropy_s15_final
TILELANG_DEFAULT_TARGET=pto python -m pytest -c /dev/null operators/cross_entropy/test_cross_entropy.py -q
```

Run Python from the same snapshot directory so that `operators.cross_entropy` resolves to these files, rather than a different snapshot:

```python
import torch
import torch_npu  # registers the NPU backend

from operators.cross_entropy.runtime import cross_entropy

logits = torch.randn(128, 32000, device="npu", dtype=torch.float32, requires_grad=True)
target = torch.randint(32000, (128,), device="npu")
loss = cross_entropy(logits, target)
loss.backward()
```

## Contract

- Positive logits shape `[N, V]`, BF16 or FP32; integer targets `[N]` on the same NPU.
- `mean`, `sum` and `none` reductions, `ignore_index`, constant class weights, label smoothing, softcap and optional z-loss regularization. A separately returned z-loss is detached for logging.
- Input logits, targets and class weights are not modified. All-ignored inputs return zero loss and zero input gradients, including for `mean` reduction.
- Class-weight gradients, model integrations, FP16 inputs and higher-order gradients are outside this standalone API.

`cross_entropy.py` contains the kernels, including the FP32 / V=32000 deferred-gradient resident path and `NORMAL_PERS` input-cache policy. `runtime.py` performs input validation, dispatch and autograd adaptation. `test_cross_entropy.py` checks forward/backward values against an independent CPU reference, feature combinations, ignored targets, vector/tile boundaries, non-contiguous storage, repeated backward, inference and invalid inputs.

## Validation and measurements

The current PR preparation uses Ascend950DT_9582 (NpuArch3510), CANN 9.2 B237, PyTorch 2.9.0, torch_npu 2.9.0.post4 and PTOAS 0.67. All 79 standalone precision tests passed on the actual PR files (0 failures, 0 errors, 0 skipped). Snapshot lint and format checks passed. Exact source identities and validation details are recorded in `source-provenance.json`.

Before the formatting-only PR cleanup, the same S15 algorithm passed 28 supported shape/dtype configurations from the upstream CI/benchmark shape matrix, including independent forward and gradient checks. On the upstream default benchmark shapes `[N,128256]`, FP32, N=1024/2048/4096/8192, the full public call was 1.403x faster than this fork's Ascend Triton path by geometric mean; inference-only forward was 1.617x. Both used NPU Events on one physical device, two counterbalanced orders, 10 warmup and 30 measured iterations per order, and a fixed 0.7 upstream gradient for training. These are complete operator-call times, including wrapper/synchronization overhead, rather than pure kernel latency or model throughput.

The original kernel/runtime hashes remain in `source-provenance.json`. PR formatting is checked for AST equivalence; the standalone tests are rerun against the actual PR files. Performance measurements are explicitly identified as pre-formatting, are specific to this DT machine, and do not establish gains for every dtype, shape, attribute combination or model. Historical PR_9579 results are not presented as current DT measurements.
