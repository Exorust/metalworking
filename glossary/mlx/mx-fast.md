# mx.fast

**`mx.fast` is MLX's drawer of fused operations (attention, norms, RoPE as single
kernels instead of op graphs) plus the escape hatch for writing your own:
`mx.fast.metal_kernel`.**

CUDA equivalent: the fused-kernel layer you'd get from cuDNN's fused attention or
FlashAttention-the-library, plus a slice of what you'd use Triton for: inline
custom kernels without leaving Python.

The fused ops (`scaled_dot_product_attention`, `rms_norm`, `layer_norm`, `rope`)
exist because of [the platform's one law](../machine/unified-memory.md): each is a
handful of graph ops fused into one kernel to avoid materializing intermediates.
[Fusion](../techniques/fusion-and-epilogues.md) as a product surface. SDPA is the
big one: it runs the [steel attention kernel](../kernels/steel-attention.md) for
prefill and the [vector kernels](../techniques/decode-vs-prefill.md) for decode.

**The dispatch gate is readable, and worth reading.** This is where "why isn't my
attention fused?" is answered:

```cpp
  const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 96 ||
       query_head_dim == 128);
  ...
  const bool supports_sdpa_full = query_sequence_length > 8 &&
      sdpa_full_supported_mask && sdpa_full_supported_head_dim;

  const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
      (query_sequence_length <= key_sequence_length) &&
      sdpa_vector_supported_head_dim &&
      (query_sequence_length * gqa_factor) <= 32;

  return !(supports_sdpa_full || supports_sdpa_vector);
```
— [`scaled_dot_product_attention.cpp:629-644`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/scaled_dot_product_attention.cpp#L629-L644), abridged

Fall outside those conditions (an unusual head dim, a mask type the kernel lacks,
training mode) and you silently get the unfused fallback graph. The
[DFlash war story's](../war-stories/three-questions.md) single best win was a
head-dim patch moving a model *inside* this gate. Note also what the file admits:
the fused kernel has no backward on Metal (training takes the fallback), one of
the sharpest open gaps in the ecosystem.

**`mx.fast.metal_kernel`** compiles an [MSL](../metal/msl.md) body from a Python
string, JIT-cached, with typed input/output specs. The platform's Triton-shaped
hole is filled by raw MSL:

```python
kernel = mx.fast.metal_kernel(
    name="my_op", input_names=["x"], output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        out[i] = 2.0 * x[i];
    """)
```

Caveats that bite CUDA people: `grid` is in
[*threads*, not threadgroups](../metal/dispatch-geometry.md); templates and
headers go in separate arguments; and there's no
[attribute or function-constant](../metal/function-constants.md) surface. For a
kernel whose performance depends on those
([register-pressure-critical GEMMs](../machine/registers.md)), drop to PyObjC and
[raw Metal](../metal/metal-the-api.md) instead. It's the right tool for fused
elementwise/reduction ops, and the wrong tool for beating steel.

Next: [Quantization](quantization.md)
