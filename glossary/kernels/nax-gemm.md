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
Source: [MLX `steel/gemm/nax.h:27-31`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/nax.h#L27-L31);
the file opens with
`#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` and
wraps MPP's tensor ops in the same loader/MMA/epilogue decomposition
[steel](../mlx/steel.md) uses everywhere else. The point to notice: the
*architecture* survived the hardware change. `BlockLoader` still stages,
the wiring kernels still wire; the MMA layer and fragment geometry
moved, and the loader (no threadgroup staging) and epilogue moved with them.

**flash-moe's standalone kernel**
([`metal_infer/nax_gemm.metal`](https://github.com/gorroai/flash-moe/blob/4df3af8278c4bef2e7f6b34f61e4e2596b58e93b/metal_infer/nax_gemm.metal),
~230 lines) shows the API without the framework: it declares an
`mpp::tensor_ops::matmul2d_descriptor` with its tile shape (32 by 32 by 32, not
the problem's M, N and K), chooses an
`execution_simdgroups<4>` scope (four cooperating simdgroups, the analogue of
a threadgroup-wide warp-specialized MMA), obtains the accumulator via
`get_destination_cooperative_tensor`, and iterates K. Its comments document
the part that bites porters: a Metal tensor lists its contiguous dimension
first (the spec fixes stride 0 at 1), so a row-major M by K matrix is described
as extents {K, M} with strides {1, K} and indexed column first, which reads as
column-major to anyone used to NumPy shapes; get the order or the transpose
flags wrong and the op silently multiplies the transpose. Apple's MPP guide
gives M5 starting points for 16-bit operands: a 2 by 2 grid of simdgroups per
threadgroup, a 32 by 32 output tile per simdgroup, BK of 128, an execution-only
barrier per K step, and a Morton-order walk over output tiles.

What distinguishes the school, against the other three case-study lineages:

- **vs [hand-placed steel](steel-gemm-fused.md)**: no `simdgroup_load`
  choreography. The descriptor states one operation's shape and the library
  executes it; the tile size, accumulator grid and lane layout (`get_coord()`,
  which hard-codes a mapping the spec calls "device specific" and offers
  `get_multidimensional_index()` to avoid assuming) are still the kernel's, so the
  [register-blocking](../techniques/register-blocking.md) choices remain. What
  you give up is the multiply itself; what you buy is the matrix unit.
- **vs [MFA codegen](mfa-codegen.md)**: specialization moves from generated
  source into descriptor parameters; the shader is generic and small.
- **vs [llama.cpp enumeration](llamacpp-attention.md)**: llama.cpp's tensor
  path coexists with its classic kernels behind runtime checks, the same
  fallback discipline MLX uses (NAX steel compiles alongside plain steel;
  dispatch picks per hardware).

Status: this school is the youngest, its tuning is visibly in
flux (recent MLX commits adjust NAX tile picks and batch limits release by
release), and the hardware it targets is M5: the API compiles on earlier
Metal 4 GPUs but llama.cpp measured no gain there. Read it as the direction
of travel, with the [tiled GEMM](gemm-tiled.md) still the foundation course.

Next section: [War stories](../war-stories/three-questions.md)
