# The Failures

**The most instructive documents on the platform are the published negative
results. Three of them, and the shared lesson: your custom kernel will probably
lose to stock, and the loss modes are predictable.**

**DFlash's reverted kernels.** A speculative-decoding port
([thread](https://www.reddit.com/r/LocalLLaMA/comments/1simszl/)) that reached
3.34× with bit-identical output, and whose repo deliberately contains **no
`.metal` files**. Every hand-written kernel tried along the way (batched GEMV,
fused SiLU, custom SDPA) benchmarked at **0.5-0.8× stock
[MLX steel](../mlx/steel.md)** and was reverted. The wins that survived were
[question-2 and question-3 moves](three-questions.md): a head-dim patch onto
[the fused-SDPA path](../mlx/mx-fast.md), and
[one `mx.eval` per step](../mlx/lazy-evaluation.md). Multiple independent
projects report the same 0.5-0.8× experience; treat it as the prior.

**OpenEvolve's evolved kernel.** An LLM-driven evolutionary search over Metal
source for Qwen3 GQA attention
([thread](https://www.reddit.com/r/LocalLLaMA/comments/1lm98z7/)), whose own
analysis file, titled "Why Optimization Failed", reports the best evolved
kernel at **3.2% slower than the MLX baseline**. The search improved from -11.5%
to -3.2% and never crossed zero. Kept as a methodology study: search can climb a
hill efficiently and still be on the wrong hill when the baseline is
[a hand-tuned library](../mlx/steel.md) and the search can't touch
[the dispatch layer above it](../mlx/how-an-op-becomes-a-kernel.md).

**"I beat MLX's SDPA... then discovered it was useless."** A hand-rolled
FlashAttention ([writeup](https://medium.com/@rajveer.rathod1301/i-built-a-flashattention-kernel-that-beat-mlxs-sdpa-then-i-discovered-it-was-useless-e4ce6ebf953c))
showing 1.2-1.3× on synthetic benchmarks and **0% end to end**: the kernel's
share of real runtime was too small for its speedup to register. The
[roofline page's hygiene rules](../techniques/roofline.md) exist because of this
failure mode; it is the most common one in public benchmark claims.

**The mechanism, finally measured.** The MTPLX project ported every kernel
from a Metal optimization challenge onto a new model's geometry and published
the full ledger
([PORT_KERNEL_LEDGER](https://github.com/youssofal/MTPLX/blob/ed1c8eea501689b744c13bec6a99ee2d36d26ab5/docs/laguna-mlxfast-port/PORT_KERNEL_LEDGER.md),
Apache-2.0): 15 kernels, GPU-validated, individually timed. Score: 2 wins,
11 losses, 1 no-lever, and both wins were *prefill fusions* where stock MLX
has no MMA-backed path. The ledger's own thesis: cross-op **fusion**
transfers; hand **GEMV/attention** kernels do not, "because they race MLX's
`mlx::steel` MMA (`gather_qmm`, flash-SDPA) which `mx.fast.metal_kernel`
cannot reach." That is the 0.5-0.8× prior explained: the fast paths live
below the API your custom kernel is written against. Wins happen where no
fast path exists to race.

The compiled checklist for reading (or making) an "X% faster on Apple Silicon"
claim:

1. Synthetic or end-to-end? ([Amdahl eats kernels](../techniques/roofline.md).)
2. Which [question](three-questions.md) did it answer? "None, I wrote a better
   kernel" predicts reversion.
3. What's the baseline: stock [steel](../mlx/steel.md)/llama.cpp at the right
   shapes, or a strawman?
4. [Prefill or decode](../techniques/decode-vs-prefill.md)? Wins in one are
   invisible in the other.
5. Thermals controlled? ([Sustained ≠ first-second](cheap-tricks.md).)

Next: [Cheap tricks](cheap-tricks.md)
