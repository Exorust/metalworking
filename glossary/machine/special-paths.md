# Special Paths

**Two hardware asymmetries that reshape kernels on this platform: a fast `exp2`
path, and float atomics that exist in the language but are slow.**

## The fast exp2

CUDA equivalent: the SFU's `MUFU.EX2`. Both platforms exponentiate in base 2 at
hardware speed; what differs is how much the local kernels lean on it. Every
attention implementation in this glossary computes softmax **in base 2**: fold
`log₂e` into the scale factor once, then use `fast::exp2` in the hot loop.

```cpp
  const AccumType scale = params->scale * M_LOG2E_F;
```
Source: [MLX `steel_attention.h:166`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L166)

This is the same math as `exp` with one fewer multiply per element. The
specification does not promise which hardware unit runs it; what it states is
accuracy: the fast single-precision `exp2` is within 3 + floor(|2x|) ulp,
against 4 ulp for the precise variant, so its error grows with the magnitude of
the argument. In a default build (fast math is Metal's default, see
[MSL](../metal/msl.md)) a plain `exp2` already selects the fast variant;
`fast::exp2` makes the choice explicit and survives a `-fno-fast-math` build. See
[online softmax](../techniques/online-softmax.md) for where this lands in the
algorithm.

## Slow FP32 atomics

CUDA equivalent: `atomicAdd(float*)` has been a cheap hardware instruction since
Kepler, and CUDA kernels use it casually for cross-block accumulation, gradient
reduction, histogram bins. **Unlearn that here.** Metal has had `atomic_float`
with `atomic_fetch_add_explicit` since Metal 3 (GPU family Apple7, so every
M-series chip), but only in device memory until Metal 4.1 added threadgroup
memory, and only with relaxed ordering. Apple says nothing about how the hardware
implements it; community measurements report it far slower than integer atomics
(consistent with a compare-and-swap loop), slow enough to dictate architecture:

- [metal-flash-attention](../kernels/mfa-codegen.md) splits its backward pass into
  two kernels, dQ in one and dK/dV in the other, *specifically* so that no output
  ever needs atomic accumulation from multiple threadgroups.
- Reduction-shaped problems prefer the two-pass pattern: partial results to a
  scratch buffer, then a small combine kernel. See the
  [decode attention kernels](../techniques/decode-vs-prefill.md), where llama.cpp
  splits the KV cache across simdgroups and merges partial softmaxes in a second
  kernel rather than accumulating atomically.

Integer atomics exist and are usable; it's specifically 32-bit float atomics that
are a trap. If your port from CUDA contains `atomicAdd` on floats in a hot path,
that's the first thing to redesign, usually into a
[split-and-combine](../techniques/decode-vs-prefill.md) or a
[fusion](../techniques/fusion-and-epilogues.md) that makes the accumulation local
to one simdgroup, where a `simd_sum` does it in registers.

Next: [AMX](amx.md)
