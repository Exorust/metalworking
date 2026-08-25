# Case Study: llama.cpp Attention

**One 11,603-line shader, `ggml-metal.metal`, serves every model architecture
and quantization format llama.cpp runs on Apple hardware. Its flash attention is
a kernel *family* around one shared implementation, specialized by explicit
enumeration: the production school.**

The family: a pad kernel, a mask-metadata precompute, the shared templated
implementation (`kernel_flash_attn_ext_impl`,
[~line 6431](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L6431)),
the [simdgroup-matrix](../metal/simdgroup-matrix.md) main path, and the
[decode vec path](../techniques/decode-vs-prefill.md). Behavior toggles via
[function constants](../metal/function-constants.md):

```metal
constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];
```
— [`ggml-metal.metal:6385-6389`](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L6385-L6389)

Where [MFA generates](mfa-codegen.md) and [MLX templates](steel-attention.md),
llama.cpp **enumerates**: an explicit `[[host_name]]` instantiation per head-dim
× dtype combination the ecosystem's models actually use.

```metal
template [[host_name("kernel_flash_attn_ext_f32_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4, 1, dequantize_f32, float4x4, 1, dequantize_f32,  64,  64>;
...
template [[host_name("kernel_flash_attn_ext_f32_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4, 1, dequantize_f32, float4x4, 1, dequantize_f32, 576, 512>;
```
— [`ggml-metal.metal:7128-7143`](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L7128-L7143), abridged

Read the template arguments: `dequantize_f32` here, and in sibling
instantiations, dequantizers for Q4, Q8, and the rest of the
[K-quant menagerie](../mlx/quantization.md). That's the school's defining
constraint: **K and V arrive quantized, so dequantization is fused into the
attention loop itself**. The asymmetric entries (`dk576_dv512`) exist for
specific model families; enumeration means the shape list *is* the ecosystem's
model list, maintained by hand.

Decode is the family's most platform-shaped member:
`kernel_flash_attn_ext_vec`
([~line 7296](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L7296))
splits the KV cache across simdgroups and merges partial softmaxes in a second
kernel (`_vec_reduce`) via the
[online-softmax correction identity](../techniques/online-softmax.md): the
[two-pass shape that emulated atomics force](../machine/special-paths.md).

History worth carrying: [PR #2615](https://github.com/ggml-org/llama.cpp/pull/2615)
is where llama.cpp **deleted [MPS](../metal/mps.md)** for these hand-written
kernels. ~88% ALU on a 4096² matmul, honestly caveated at ~40% end-to-end
because non-matmul ops dominate. The quantized GEMM family shares the file
(`kernel_mul_mm`,
[~line 10048](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L10048));
most [war stories](../war-stories/sparse-v.md) are fought inside it.

Next: [The NAX GEMM](nax-gemm.md)
