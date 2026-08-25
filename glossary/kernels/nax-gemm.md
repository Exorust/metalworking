# Case Study: The NAX GEMM

**The fourth kernel school: GEMM through [Metal 4 tensor
ops](../metal/mtltensor-and-mpp.md) on the [M5 neural
accelerators](../machine/neural-accelerators.md). Where the other schools
hand-place every fragment, this one describes the matmul and lets the
library and hardware place it.**

Two implementations sit in the fetched repos, and reading them together is
the lesson.

**MLX's NAX steel** (`steel/gemm/nax.h`, MIT) rebuilds the
[BlockMMA](steel-blockmma.md) idea on 16×16 fragments:

```cpp
struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;
```
— [MLX `steel/gemm/nax.h:27-31`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/nax.h#L27-L31);
the file opens with
`#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and
wraps MPP's tensor ops in the same loader/MMA/epilogue decomposition
[steel](../mlx/steel.md) uses everywhere else. The point to notice: the
*architecture* survived the hardware change. `BlockLoader` still stages,
the wiring kernels still wire; only the MMA layer and fragment geometry
moved, which is exactly what the decomposition promised.

**flash-moe's standalone kernel**
([`metal_infer/nax_gemm.metal`](https://github.com/gorroai/flash-moe/blob/4df3af8278c4bef2e7f6b34f61e4e2596b58e93b/metal_infer/nax_gemm.metal),
~230 lines) shows the API without the framework: it declares an
`mpp::tensor_ops::matmul2d_descriptor` with the problem shape, chooses an
`execution_simdgroups<4>` scope (four cooperating simdgroups, the analogue of
a threadgroup-wide warp-specialized MMA), obtains the accumulator via
`get_destination_cooperative_tensor`, and iterates K. Its comments document
the part that bites porters: the tensor ops think in column-major layouts,
and getting row-major LLM weights through them is layout gymnastics that the
descriptor does not hide.

What distinguishes the school, against the other three case-study lineages:

- **vs [hand-placed steel](steel-gemm-fused.md)**: no `simdgroup_load`
  choreography, no per-fragment stride constants. The descriptor states the
  shape; placement is the library's problem. That surrenders the
  [register-blocking](../techniques/register-blocking.md) dials this
  glossary spends pages on, and buys the new hardware's throughput.
- **vs [MFA codegen](mfa-codegen.md)**: specialization moves from generated
  source into descriptor parameters; the shader is generic and small.
- **vs [llama.cpp enumeration](llamacpp-attention.md)**: llama.cpp's tensor
  path coexists with its classic kernels behind runtime checks, the same
  fallback discipline MLX uses (NAX steel compiles alongside plain steel;
  dispatch picks per hardware).

Status, honestly: this school is the youngest, its tuning is visibly in
flux (recent MLX commits adjust NAX tile picks and batch limits release by
release), and nothing here runs on pre-M5 hardware. Read it as the direction
of travel, with the [tiled GEMM](gemm-tiled.md) still the foundation course.

Next section: [War stories](../war-stories/three-questions.md)
