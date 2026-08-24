# Case Study: The Fused GEMM Kernel

**`steel_gemm_fused.h` is what remains of a GEMM once
[BlockLoader](steel-blockloader.md), [BlockMMA](steel-blockmma.md), and
[epilogues](../techniques/fusion-and-epilogues.md) are factored out: ~350 lines of
wiring, most of which is [function-constant](../metal/function-constants.md)
dispatch. The kernel is the least interesting file in the library, which is the
achievement.**

The head declares both specialization mechanisms at once: template parameters
(shape; fixed per compiled variant) and function constants (behavior; fixed per
pipeline).

```cpp
constant bool has_batch [[function_constant(10)]];
constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

template <
    typename T,
    int BM, int BN, int BK,
    int WM, int WN,
    bool transpose_a, bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm(
```
— [`steel_gemm_fused.h:9-29`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L9-L29), reformatted

Note [`max_total_threads_per_threadgroup`](../machine/registers.md) as standard
practice, and which
[tile parameters the host picks per shape](../mlx/how-an-op-becomes-a-kernel.md).

The aligned fast path is the [m5-gemm K-loop](gemm-tiled.md) rewritten against
component interfaces, containing zero bounds checks because
[pipeline-time specialization deleted them](../metal/function-constants.md):

```cpp
  if (align_M && align_N) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    ...
    return mma_op.store_result(D, params->ldd);
```
— [`steel_gemm_fused.h:172-204`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L172-L204), abridged

The unaligned branches below
([lines 209-345](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L209-L345))
enumerate every ragged-edge combination through `load_safe`/`store_result_safe`:
verbose, mechanical, and *absent from aligned pipelines*.

Once this file reads as boring, the rest of the
[steel](../mlx/steel.md) shape space is a names tour: `steel_gemm_splitk.h`
(huge-K: split the reduction across threadgroups, combine after),
`steel_gemm_masked.h` (block-sparse), `steel_gemm_gather.h` (MoE gather-GEMM),
`steel_gemm_segmented.h`. All reuse the same loader/MMA components with
different wiring. That reuse is the whole argument for the decomposition, and the
`_nax` twins (`gemm_nax.h`) isolate exactly what the Metal-4/M5 tensor
instructions change: diff one against its plain sibling and nothing moves but the
MMA layer.

Next: [Steel attention](steel-attention.md)
