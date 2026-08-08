# Stage 3 — How production factors it

**Needs:** `code/mlx-steel-kernels`, `code/luminal` (optional) &nbsp;·&nbsp; **Time:** 2–3 days

Stage 2 gave you a fast GEMM as one hand-written file. Production libraries can't work
that way — they need every dtype, shape, epilogue, and quantization scheme from one
codebase. MLX's **steel** library is the CUTLASS-style answer: the GEMM you already
understand, decomposed into composable templated pieces. Your stage-2 knowledge is the
key that makes this code readable.

Root: `code/mlx-steel-kernels/mlx/backend/metal/kernels/`.

> **Orientation warning:** there are **two** independent steel subsystems — `steel/gemm/`
> and `steel/attn/` — each with its own `mma.h`/`loader.h` pair and *different* fragment
> layouts. Do not mix them up. You read `gemm/` now; `attn/` is stage 4.

## Read

- `steel/gemm/loader.h` — `BlockLoader<T, BROWS, BCOLS, dst_ld, reduction_dim, tgp_size,
  alignment>`: the vectorized global→threadgroup cooperative loader. The read-count /
  row-count arithmetic is derived at compile time from the threadgroup size — this is
  your stage-2 cooperative load, generalized.
- `steel/gemm/mma.h` — `BlockMMA`: the register-tile MMA layer built on
  `simdgroup_matrix` accumulator fragments. The core "how to tile registers on Apple
  silicon" file.
- `steel/gemm/kernels/steel_gemm_fused.h` — GEMM with fused epilogue (bias/activation):
  lesson 1 (fusion beats extra passes) as production code. Skim the siblings to see the
  shape space one template family covers: `steel_gemm_splitk.h` (split-K),
  `steel_gemm_masked.h` (block-sparse), `steel_gemm_gather.h` (MoE gather-GEMM),
  `steel_gemm_segmented.h`.
- `quantized.h` — MLX's group-wise affine quantized matmul/QMV kernels. Stage 6's war
  stories mostly fight this file.
- The `*_nax.h` / `*_nax.metal` variants (`gemm_nax.h`, `fp_quantized_nax.metal`) are
  the newer-hardware tensor-op path. Diff one against its non-`nax` sibling to see
  exactly what the new instruction set changes.

## Optional deep end: the compiler school

Luminal generates its kernels instead of templating them —
[Compiling fast GPU kernels](https://docs.luminalai.com/blog/gpu) explains the approach
(elementwise fusion, one command buffer for all of Llama 3 8B, search-based compilation
that rediscovers flash attention).

- `code/luminal/crates/luminal_metal/src/kernel/ops.rs` — 3468 lines, the only file in
  the crate containing MSL. Each op is a struct emitting a `kernel void mkernel(...)`
  string with shapes/strides/dtypes substituted at compile time (`GenericMatmul` at 2065,
  its MSL at ~2236). Fusion is expressed as e-graph rewrites, which makes the fused-kernel
  set open-ended rather than a fixed file list.
- `code/luminal/crates/luminal_metal/src/memory_analysis.rs` — 1478 lines of static
  buffer-lifetime analysis: compile-time allocation reuse, ~zero allocations per forward
  pass. The interesting non-kernel systems work.

## Done when

- You can map each stage-2 concept to its steel home: cooperative load → `BlockLoader`,
  register accumulation → `BlockMMA`, tile shape → which template parameters.
- You can explain why `attn/` needs its own `mma.h` instead of reusing `gemm/`'s.
- You can say what a fused epilogue saves, in terms of lesson 1.

**Next:** [Stage 4 — Flash Attention, three ways](stage-4-attention.md)
