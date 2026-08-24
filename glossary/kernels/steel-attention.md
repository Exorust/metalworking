# Case Study: Steel Attention

**`steel_attention.h` (~476 lines) is the complete
[flash attention](../techniques/flash-attention.md) forward built from the
[steel vocabulary](../mlx/steel.md): the most readable production FA on the
platform, and the payoff for knowing [online softmax](../techniques/online-softmax.md)
before you arrive.**

Specialization first: the [function-constant](../metal/function-constants.md)
set that compiles masking and raggedness out of pipelines that don't need them.

```cpp
constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];
constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
```
— [`steel_attention.h:11-16`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L11-L16)

The register layout tells you the loop's shape: Q resident, K/V streaming, and
`Stile` (the score tile that naive attention writes to DRAM) as pure
[register](../machine/registers.md) state.

```cpp
  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
```
— [`steel_attention.h:186-190`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L186-L190)

`Stile` is born from `Q·Kᵀ`, masked in place, softmaxed in place, multiplied
against V, and dies without ever touching memory. This in-place transformation is
why `attn/` [forks its own `mma.h`](steel-blockmma.md) with a different fragment
layout.

Then the [derivation](../techniques/online-softmax.md), line for line, inside the
KV loop (`ExpSubOp::apply(x,y) = fast::exp2(x - y)`; base-2 via `scale *=
M_LOG2E_F` at [line 166](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L166)):

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
— [`steel_attention.h:391-420`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L391-L420), abridged

`factor` is the correction `c`; the final `DivOp` normalize is at
[line 460](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L460).
If the [online-softmax page](../techniques/online-softmax.md) landed, this file
holds no surprises, which is the point of reading it second.

Details that reward attention: causal handling is a **loop-bound computation, not
a mask**. `kb_lim` at
[lines 239-247](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L239-L247)
means tiles above the diagonal are never visited
([question 1: delete work](../war-stories/three-questions.md)). K loads
transposed via a differently-parameterized
[BlockLoader](steel-blockloader.md). GQA is a stride trick (`kv_head_idx =
tid.y / gqa_factor`). What this kernel serves:
[`mx.fast.scaled_dot_product_attention`'s prefill path](../mlx/mx-fast.md), for
the head dims its [dispatch gate](../mlx/mx-fast.md) accepts. Decode goes
[elsewhere](../techniques/decode-vs-prefill.md); backward
[doesn't exist on Metal](../techniques/flash-attention.md).

Next: [metal-flash-attention, the codegen school](mfa-codegen.md)
