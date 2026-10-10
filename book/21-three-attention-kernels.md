# Three attention kernels, one algorithm

## Where we are

Chapter 20 derived online softmax and the flash-attention shape. Three production
kernels on the platform implement it. The algorithm is the same in all three, so
it is the least interesting thing about them. Reading them together shows where
they disagree on engineering: how each one gets its specialization, how each one
spends registers, and what each one is shaped by.
This chapter reads MLX's kernel in the most detail, because it is the most
readable and the one the running example runs on, then metal-flash-attention and
llama.cpp for their differences, and ends with the gate that decides whether
your attention runs on any of them.

## MLX: steel attention

`steel_attention.h`, about 476 lines, is the complete flash-attention forward
built from the steel vocabulary of Chapter 18, and having done Chapter 20 first
is what makes it readable. The file opens with specialization: the function-constant
set that compiles masking and raggedness out of pipelines that do not need them.

```cpp
constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];
constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
```
Source: [`steel_attention.h:11-16`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L11-L16)

The register layout tells you the loop's shape before you read the loop: Q
resident, K and V streaming, and `Stile`, the score tile that naive attention
writes to memory, as pure register state.

```cpp
  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
```
Source: [`steel_attention.h:186-190`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L186-L190)

`Stile` is born from `Q * K^T`, masked in place, softmaxed in place, multiplied
against V, and dies without touching memory. That in-place transformation is why
the attention directory has its own multiply component with a different
fragment layout, Chapter 18. The key-value loop then contains the derivation of
Chapter 20, line for line. `ExpSubOp::apply(x, y)` is `fast::exp2(x - y)`, and
the base-2 fold `scale *= M_LOG2E_F` is at line 166, the excerpt of Chapter 5.

```cpp
    // Row max
    Stile.template row_reduce<MaxOp>(new_max);
    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);
    // Factor exp(rowmax(Si) - rowmax(Si-1))
    for (short i = 0; i < kRowsPT; ++i)
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    ...
    // Update norm
    sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    // Update O
    Otile.template row_bin_op<MulOp>(factor);
```
Source: [`steel_attention.h:391-420`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L391-L420), abridged

`factor` is the correction `c`, and the final division by the sum is at line 460.
If Chapter 20 landed, this file holds no surprises, which is the point of
reading it second. A few other details are worth carrying. Causal handling is a loop-bound
computation, not a mask; `kb_lim` at lines 239 to 247 means tiles above the
diagonal are never visited. K loads transposed through a differently
parameterized `BlockLoader`. Grouped-query attention is a stride trick,
`kv_head_idx = tid.y / gqa_factor`. This kernel serves the prefill path
of `mx.fast.scaled_dot_product_attention`, for the head dimensions its gate
accepts; decode goes elsewhere, Chapter 22, and backward does not exist on Metal,
Chapter 20.

## metal-flash-attention: the codegen school

philipturner's metal-flash-attention reached 83 percent arithmetic-unit
utilization on an M1 Max, and its structural bet shows in one fact: there
are no `.metal` files in the repository. Every kernel is assembled as a Swift
string for the exact problem shape, then compiled at run time through the path
of Chapter 6.

```swift
    return """
    \(createMetalSimdgroupEvent())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;

    \(createConstants())

    // Declare the function.
    kernel void attention(
      \(createBufferBindings())
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
      ...
      \(createSetup())
      \(createLoop())
      \(createCleanup(type: type))
    }
    """
```
Source: [`AttentionKernel+Source.swift:23-54`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Source.swift#L23-L54), abridged

Where MLX uses templates and llama.cpp enumerates, metal-flash-attention writes a
bespoke kernel: head dimension, precision, transpose state, even loop bounds
become literals. The result is maximum specialization with no shipped binary.
Three positions distinguish the project. First, it spills on purpose: it adds a
third blocking dimension along the head dimension, with tile shapes sometimes
chosen to spill predictably, on the thesis that a mapped spill of the right
operand beats a smaller tile. Given Chapter 2's warning, this is the boldest
register-pressure stance of any kernel in this book. Second, it has a backward
pass, the only open-source one on the platform,
split into dQ and dK, dV kernels because of the emulated float atomics of
Chapter 5. Third, its tile staging is built on the dead copy intrinsics of Chapter
14, with the comment documenting the M1-era hang. The repository is pre-Metal 4
and unmaintained since 2024, but its production descendant, MFA 2.0 in C++
inside the Draw Things application, ships int8 and sparse-indexed attention to
real users. For the engineering narrative behind decisions like these, llama.cpp
pull request 5021, where the flash-attention kernel was built across 154 public
comments, is the best line-by-line account anywhere.

## llama.cpp: the enumeration school

One shader file of 11,603 lines, `ggml-metal.metal`, serves every model
architecture and quantization format llama.cpp runs on Apple hardware. Its flash
attention is a family around one shared implementation, specialized by explicit
enumeration: an instantiation per head-dimension and data-type combination the
ecosystem's models use, each with its own host name. The list of shapes follows
the list of models and is maintained by hand.

```metal
template [[host_name("kernel_flash_attn_ext_f32_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  32,  32>;
...

```
Source: [`ggml-metal.metal:7128-7143`](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L7128-L7143), abridged

Read the template arguments: `dequantize_f32` here, and in sibling instantiations
the dequantizers for the quantized formats of Chapter 23. This school is defined
by one constraint: K and V arrive quantized, so dequantization is fused into the
attention loop itself. The asymmetric entries such as `dk576_dv512` exist for
specific model families. Behavior toggles ride on function constants, Chapter 8,
and its decode member, the vector kernel, is the subject of Chapter 22. This file
is also where llama.cpp deleted MPS in 2023, Chapter 9, and most of the measured
wins in Part 7 were made inside it.

## The disagreements, named

The three implementations share one algorithm and differ on three axes.

| Axis | MLX | metal-flash-attention | llama.cpp |
|---|---|---|---|
| How a shape gets its kernel | templates plus function constants | generated source per shape | one instantiation per shape, by hand |
| Register pressure | avoid spills | spill on purpose where mapped | fixed by the enumerated shape |
| Correction step | unconditional multiply | guarded by `if (m_new > m)` | `exp` on the shifted score, lowering left to the compiler |

On the same hardware with the same mathematics, all three are defensible. Once
the algorithm is settled, the remaining decisions are measured trade-offs, and
the measurements in Part 7 settle them.

## The gate: why is my attention not fused?

`mx.fast` is MLX's drawer of fused operations, attention, normalizations and
rotary embedding as single kernels, plus the escape hatch `mx.fast.metal_kernel`
for writing your own. Its attention entry runs steel attention for prefill and
the vector kernels for decode, and the decision between fused and fallback is
readable.

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
Source: [`scaled_dot_product_attention.cpp:629-644`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/scaled_dot_product_attention.cpp#L629-L644), abridged

If your model falls outside those conditions, with an unusual head dimension, a
mask type the kernel lacks, or training mode, then at the pinned commit you
silently got the unfused fallback graph. Current MLX removes the silence: the gate is refactored into
`has_fused_kernel()`, which returns a human-readable reason, and a `force_fused`
argument raises with that reason instead of falling back. The DFlash case of
Chapter 24 got its single best kernel-level win from a head-dimension patch that
moved a model inside this gate.

`mx.fast.metal_kernel` compiles a Metal Shading Language body from a Python
string, cached, with typed input and output specifications. Three caveats catch
CUDA programmers: its `grid` is in threads, not threadgroups, Chapter 7; templates and
headers go in separate arguments; and there is no attribute or function-constant
surface, so a kernel whose performance depends on those, the register-critical
matrix multiplies of Part 4, needs PyObjC and raw Metal instead. It is the right
tool for fused elementwise and reduction operations and the wrong tool for
beating steel, which the MTPLX ledger of Chapter 25 measured.

## Where the matmul stands

The attention example runs on steel attention for prefill, with the function
constants that match its shape, and only if its head dimension and mask type
pass the gate. One query row at a time, the generation case, takes a different
kernel, and the next chapter is about why.

## Check yourself

1. A model has a head dimension of 112 and runs on the pinned MLX commit. What
   happens to its attention, and how would you find out on current MLX?
2. Name one engineering decision on which metal-flash-attention and MLX disagree,
   and say what each side is trading.
3. Why does llama.cpp's attention take a dequantizer as a template argument?

### Answers

1. 112 is not in the supported list, so the gate returns true and the attention
   runs as the unfused fallback graph, silently. On current MLX,
   `has_fused_kernel()` returns the reason, and `force_fused` raises with it.
2. The correction step: metal-flash-attention guards it with `if (m_new > m)`,
   trading a branch for a skipped multiply when the maximum did not move, and MLX
   multiplies unconditionally, trading a multiply for no divergence. Or register
   pressure: metal-flash-attention spills on purpose for a larger tile, MLX
   avoids spills.
3. Because its K and V arrive quantized, so each instantiation dequantizes inside
   the attention loop with the format's own routine, rather than reading a
   dequantized copy that would cost the traffic the quantization saved.

Every kernel in this chapter stops working when the query is one row, and the
next chapter is about what runs instead and why its speed is a bandwidth number.

Next: [The KV cache, decode, and prefill](22-the-kv-cache-decode-and-prefill.md)
