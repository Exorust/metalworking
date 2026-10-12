# The two asymmetries: fast exp2, emulated float atomics

## Where we are

Chapters 2 to 4 gave you the chip as a set of budgets: registers, scratch memory,
bandwidth, resident simdgroups. You still need two facts about the arithmetic that
do not fit into any budget, and both change the shape of kernels. One is an exponent path that the
attention kernels lean on. The other is an operation CUDA code uses casually and
this hardware runs slowly. Part 1 ends with them because the next three parts
assume you know both.

## The fast exp2

If both platforms exponentiate in base 2 at hardware speed, what is there to
learn? On NVIDIA the special function unit has an instruction for it, `MUFU.EX2`,
and on Apple the Metal Shading Language function `fast::exp2` is the direct route
to the equivalent path. What differs is how much the local kernels lean on it.

Softmax is the operation that needs an exponent per element. The attention
kernels in MLX and in metal-flash-attention compute it in base 2, using a one-line
identity:

    exp(x) = exp2(x * log2(e))

Attention already multiplies every score by a scale before the exponent, so the
constant `log2(e)` folds into that scale once, outside the loop, and the hot loop
calls `fast::exp2` on the scaled score. Here is the fold in MLX's attention kernel:

```cpp
  const AccumType scale = params->scale * M_LOG2E_F;
```
Source: [MLX `steel_attention.h:166`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L166)

The point of the fold is not the multiply, which a compiler hoists out of the
loop anyway; it is that the loop then calls `fast::exp2`, the hardware exponent,
instead of whatever a library `exp` expands to.

Once you scale the scores into base-2 units, the running maximum and correction
terms must follow. A stable softmax subtracts that maximum and keeps those
terms, so the correction becomes `fast::exp2(m2_old - m2_new)`, and an additive
mask or bias has to be multiplied by `log2(e)` (not by the folded scale), which
the same MLX kernel does.

`fast::exp2` is also a reduced-precision path, like CUDA's `__exp2f`, so the
first time you apply it to a kernel, check the output against the original.
Chapter 20 derives the
algorithm the fold lives in. Not every production kernel does it: the pinned
llama.cpp attention calls `exp` on the shifted score and leaves the lowering to
the compiler.

## Float atomics that run slowly

In CUDA, `atomicAdd` on a 32-bit float is a hardware instruction on every
current GPU, and CUDA kernels use it freely: accumulation across
thread blocks, gradient reductions, histogram bins. Do not carry that habit over.
The Metal API accepts float atomic adds, but on Apple hardware they are emulated
with compare-and-swap loops. The emulation costs more than a few percent; it is
slow enough to decide how a kernel is structured.

Chapter 1 said that two threadgroups in one grid can combine results through
atomic operations on device memory. For floats, that route is legal and slow, and
the attention kernels in this book are shaped around avoiding it. MLX's general
reductions do still call the emulated add on their outputs, where contention is
rare; the compare-and-swap loop you will find written out by hand in MLX's
`atomic.h` is for float min and max, for which no builtin exists at all.

So how do those kernels avoid it? metal-flash-attention splits its backward
pass into two kernels, one that produces dQ and one that produces dK and dV. Each
output element then has one owning threadgroup: the dQ kernel owns tiles of query
rows and walks the keys, the dK and dV kernel owns tiles of key rows and walks the
queries, at the price of recomputing the scores in both. No element is ever
accumulated by two threadgroups. Chapter 21 reads that kernel.

When several threadgroups contribute to one sum, give each its own partial
result and add a combine step instead. The
decode attention kernel of Chapter 22 is the example: llama.cpp splits the
key-value cache across simdgroups and across threadgroups, merges the
simdgroups' partials inside the first kernel through threadgroup memory, and
merges the threadgroups' partials in a second kernel. Two partial softmaxes
cannot be summed directly. Each carries its own maximum and denominator and is
rescaled to a common maximum before the sum, which is the correction that
Chapter 20 derives.

Integer atomics are native and usable; the problem is limited to 32-bit float
atomics. When your port from CUDA contains `atomicAdd` on floats in a hot path, that
is the first thing to redesign, and there are two directions to take it.

1. **Split and combine.** Give each threadgroup its own slice of the output, write
   partials to a scratch buffer, and combine them in a second dispatch. For P
   32-bit partial values this costs 4P bytes written and 4P bytes read, plus one
   extra dispatch; the decode kernel above also writes a maximum and a sum per
   partial.
2. **Make the accumulation local to one simdgroup.** When every contribution to a
   given sum can be arranged to live in one simdgroup, `simd_sum` combines them in
   registers with no memory traffic for the intermediate. Fusion, Chapter 15, is
   often what makes this arrangement possible.

The second option moves the fewest bytes, and it also puts the whole sum on one
simdgroup, which can cost parallelism. The sixty-second arithmetic of Chapter 3
says which direction to try first; a measurement says which one wins.

## Why these two belong together

What do these two change in the attention kernel you would bring over from
CUDA? Both are facts about
the arithmetic units that the budgets of Chapters 2 to 4 do not capture, and both show up in the attention kernels of Part 6. The exponent
fold is the one arithmetic trick every attention kernel on the platform shares.
The slow float atomic removes the easy way to combine results across
threadgroups, and each of the three workarounds has a price: owner-computes
recomputes the scores, split-and-combine writes and rereads the partials and
adds a dispatch, and `simd_sum` stays in registers only when the whole sum fits
in one simdgroup. Which price is smallest is a byte count. You do Chapter 3's arithmetic, and then
you measure.

That is Part 1. You now have the vocabulary of the two hierarchies, the budgets
inside a core, the law that says most kernels are bandwidth bound and the ridge
point that quantifies it, the occupancy constants and the case for 16-bit values,
and the two instruction-level asymmetries.

## Review the content

1. A CUDA softmax kernel computes `exp(s * scale - m)` for each score `s` with a
   running maximum `m`. Rewrite the computation the way the Metal attention
   kernels do, and say what else has to change for the result to be the same.
2. A ported kernel accumulates a per-row sum with `atomicAdd` on floats from
   several threadgroups that each own a slice of the row. What are the two
   redesigns the chapter offers, and which one moves fewer bytes?
3. Why does metal-flash-attention compute dQ in a different kernel from dK and dV?

### Answers

1. Compute `scale2 = scale * log2(e)` once outside the loop, keep the running
   maximum in the same scaled units, and compute `fast::exp2(s * scale2 - m2)` in
   the loop, with the correction factor as `fast::exp2(m2_old - m2_new)`. The
   gain is the hardware exponent, not the hoisted multiply. Any additive mask
   must be multiplied by `log2(e)`, and the output should be checked, because
   `fast::exp2` is a reduced-precision path.
2. Split and combine, where each threadgroup writes its partial and a second
   kernel adds them; or give each row to one simdgroup and use `simd_sum` in
   registers. The second moves fewer bytes, because the partials never leave
   registers, but it is available only when the whole sum fits in one simdgroup,
   and a measurement decides whether the lost parallelism costs more.
3. So that every output element has exactly one owning threadgroup: the dQ kernel
   owns query rows and the dK and dV kernel owns key rows. Accumulating one
   element from two threadgroups would need float atomic adds, which are emulated
   and slow on this hardware.

Part 2 leaves the hardware and asks what the least amount of Metal is that you
need to compile a kernel, launch it, and time it.

Next: [From source to pipeline state](06-from-source-to-pipeline-state.md)
