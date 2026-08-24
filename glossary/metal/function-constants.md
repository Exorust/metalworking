# Function Constants

**Function constants are values bound at pipeline-creation time that the final
compile treats as literals: Metal's mechanism for specializing one shipped kernel
into many optimized variants.**

CUDA equivalent: the *effect* of template instantiation or JIT-compiling per
configuration, without paying for either. CUDA specialization means compiling many
kernels (fatbins full of template instantiations) or NVRTC at runtime. Metal's
[compilation pipeline](compilation-pipeline.md) always finishes on-device, so it
can afford a middle path: ship *one* AIR function with declared constant slots,
bind values when building the pipeline state, and let the backend compiler fold
them, resolving branches and deleting dead code before the GPU ever sees it.

The canonical use, from MLX's fused GEMM:

```cpp
constant bool has_batch [[function_constant(10)]];
constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];
```
— [`steel_gemm_fused.h:9-16`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L9-L16)

When the host knows the matrix divides evenly by the tile (`align_M && align_N`),
the compiled pipeline contains **only the fast path**: unchecked vectorized loads,
no edge handling anywhere ([the case study](../kernels/steel-gemm-fused.md) shows
the two paths side by side). Attention kernels use the same trick for causality and
masking: [`do_causal`, `has_mask`](../kernels/steel-attention.md) are constants,
so a causal-attention pipeline carries zero masking code it doesn't need.
llama.cpp's [`FC_flash_attn_ext_*` family](../kernels/llamacpp-attention.md) is the
same pattern at production scale. This is also why simdgroup-level branch divergence
worries you less here than in CUDA: the branches that matter most are gone before
execution.

The division of labor among Metal's three configuration channels:

- **Templates / `-D` defines**: things that change register allocation or memory
  layout (tile shapes, dtypes). Fixed before AIR.
- **Function constants**: behavioral flags and occasionally values (llama.cpp
  passes stride ints). Fixed at pipeline creation. One AIR function, many
  pipelines; pipelines are cached per constant-combination.
- **Buffer arguments**: true runtime values (sizes, pointers).

A related contract lives in the kernel attribute
`max_total_threads_per_threadgroup`: not a function constant, but the same
philosophy. Tell the pipeline-time compiler what the runtime will do, and
[the register allocator repays the promise](../machine/registers.md).

Next: [Command buffers](command-buffers.md)
