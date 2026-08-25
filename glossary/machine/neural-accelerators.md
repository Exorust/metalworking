# Neural Accelerators

**The M5 generation adds a dedicated matrix unit to every [GPU
core](gpu-core.md): the neural accelerators (NAX in MLX's source), reached
through [Metal 4 tensor ops](../metal/mtltensor-and-mpp.md) rather than
`simdgroup_matrix`.**

CUDA equivalent: the Tensor Core, finally for real. Earlier pages tell the
pre-M5 story on purpose: [`simdgroup_matrix`](../metal/simdgroup-matrix.md)
executes on the regular FP32 pipes, and the reverse-engineered
[Rigel](https://arxiv.org/abs/2606.12765) work confirmed no dedicated matrix
datapath as late as M4. On M5 that changed. Apple's own numbers for MLX on M5:
[~3.97× time-to-first-token and 1.19-1.27× decode over
M4](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), with
the prefill gain landing exactly where a matrix unit should land
([compute-bound](../techniques/arithmetic-intensity.md) territory; decode
stays [bandwidth-bound](unified-memory.md), so it moves with the memory spec
instead).

What a CUDA person should map over, and what not to:

- **Fragment shape moves from 8×8 to 16×16.** MLX's NAX steel is built on a
  16×16 fragment held across the 32 threads:

```cpp
struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;
```
— [MLX `steel/gemm/nax.h:27-31`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/nax.h#L27-L31),
which includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` at
the top of the file: the new hardware arrives through a library, not new
intrinsics.

- **The formats are block-scaled floats.** The
  [mxfp4/nvfp4/mxfp8 quantization modes](../mlx/quantization.md) exist because
  this unit consumes them natively; the
  [fp8-is-emulated caveat](f16.md) is an M4-era fact.
- **It is not programmed like a tensor core.** No `mma.sync` descent: you
  describe a matmul with a descriptor and cooperative tensors, and the
  compiler and hardware pick the datapath. The
  [Metal 4 tensor-ops page](../metal/mtltensor-and-mpp.md) covers the model;
  the [NAX GEMM case study](../kernels/nax-gemm.md) reads real code.

Ecosystem status, honestly: MLX ships a parallel [steel](../mlx/steel.md)
fork for it (`gemm/nax.h`, `attn/nax.h`, `quantized_nax`, `fp_quantized_nax`);
llama.cpp has a tensor-API matmul path; and the tuning is young. Recent MLX
work is disproportionately NAX-shaped (qmv batch limits raised for M5-class
GPUs, NVFP4 QMV optimization, per-expert tile picks in `gather_qmm_rhs_nax`),
which is what a fast-moving fast path looks like. `simdgroup_matrix` remains
the portable primitive and the one most shipped kernels still use; this page
is why the glossary keeps teaching both.

Next section: [Metal, the stack](../metal/metal-the-api.md)
