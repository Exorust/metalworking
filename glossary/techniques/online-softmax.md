# Online Softmax

**Online softmax computes a numerically-safe softmax in one streaming pass,
maintaining a running max, running sum, and running output that are *corrected*
as new tiles arrive. It's the algorithm that makes
[flash attention](flash-attention.md) possible.**

CUDA equivalent: the same algorithm (it's from the FlashAttention lineage and
before it, Milakov & Gimelshein). Hardware-independent math, included here
because every Apple kernel implements it and this glossary's
[case studies](../kernels/steel-attention.md) expect you to recognize it on sight.

The problem: safe softmax is `exp(x_i - m) / Σ exp(x_j - m)` where `m` is the row
max: two whole-row facts you don't have while streaming the row tile by tile.
The resolution is one identity, `exp(x - m_old) = exp(x - m_new) · exp(m_new -
m_old)`. When a new tile raises the max, every already-accumulated term is off
by exactly the factor `c = exp(m_old - m_new)`. So, per tile of scores `S`:

```
m_new = max(m, rowmax(S))
c     = exp(m - m_new)         # correction, ≤ 1
P     = exp(S - m_new)         # this tile's weights
l     = l·c + rowsum(P)        # rescale history, add new
O     = O·c + P·V_tile         # same correction, same reason
m     = m_new
```

After the last tile, `O /= l`. Five lines; being able to re-derive the correction
factor is the check that you understand it.

Recognition guide: the same five lines in two production dialects.

```cpp
    // Factor exp(rowmax(Si) - rowmax(Si-1))
    factor[i] = fast::exp2(max_score[i] - new_max[i]);
    ...
    sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    // Update O
    Otile.template row_bin_op<MulOp>(factor);
```
— [MLX `steel_attention.h:397-420`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L397-L420), abridged

```swift
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }
```
— [metal-flash-attention `AttentionKernel+Softmax.swift:293-298`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Softmax.swift#L293-L298)

(Note MFA's `if` guard vs MLX's unconditional multiply: a genuine branch-cost vs
multiply-cost disagreement between two good kernels.) Both compute in **base 2**,
with `log₂e` folded into the scale once and
[`fast::exp2` in the loop](../machine/special-paths.md).
The row reductions inside `rowmax`/`rowsum` are
[simdgroup shuffles](../machine/simdgroup.md). And the split-KV
[decode kernels](decode-vs-prefill.md) reuse the same correction identity a
second time, to merge *partial softmaxes computed by different simdgroups*,
which is how they avoid [float atomics](../machine/special-paths.md).

Next: [Flash attention](flash-attention.md)
