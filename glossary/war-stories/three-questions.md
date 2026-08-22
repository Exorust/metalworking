# The Three Questions

**Before writing any custom kernel on this platform, ask three questions in
order. Every measured win in these war stories answers one of them; nearly every
measured failure skipped them and went straight to "write a better kernel."**

1. **Can I delete work?**
2. **Can I unlock an existing fast path?**
3. **Can I cut dispatch/sync overhead?**

Only if all three come up empty does a custom kernel have decent odds — because
the default kernels ([steel](../mlx/steel.md), llama.cpp's) are good, and the
[repeated experimental result](the-failures.md) is that hand-written replacements
come back slower.

**Question 1 — delete work.** The purest specimen is the
[sparse-V dequant skip](sparse-v.md): +22.8% decode from three lines that skip
provably-negligible work. Same family: [steel attention's causal
handling](../kernels/steel-attention.md) never visits tiles above the diagonal;
vocabulary compaction trims the output matmul for constrained decoding.

**Question 2 — unlock a fast path.** The stack is
[full of dispatch gates](../mlx/how-an-op-becomes-a-kernel.md) — head-dim lists,
sequence-length thresholds, tile-size lookup tables — and falling off a fast path
is silent. The measured wins:

- **DFlash** (speculative decoding port, 3.34× total): every hand-written custom
  kernel in the project lost to stock MLX and was reverted; the real kernel-level
  win was a head_dim patch moving the model *inside*
  [the fused-SDPA gate](../mlx/mx-fast.md).
- **MTPLX** (multi-token prediction, 2.24×: 28 → 63 tok/s): stock
  [quantized matvec](../mlx/quantization.md) stalls at the M=3-6 shapes
  speculative decode produces — between the M=1 tuning and the GEMM tiles. ~10
  lines of Metal (wider simdgroups plus unrolling, bit-exact) fixed it. A
  textbook question-2 result: the kernel existed; its tuning grid had a hole.

**Question 3 — cut dispatch/sync overhead.** The
[platform tax](../metal/command-buffers.md) is real and the fixes are cheap:
DFlash halved GPU→CPU syncs with a single
[`mx.eval()` per step](../mlx/lazy-evaluation.md); **flash-moe** (209 GB MoE at
20 tok/s in 128 GB RAM, pure C/ObjC/Metal) pre-encodes command buffers and
prefetches experts during GPU compute on the observation that ~27% of experts
recur token-to-token — its thread lists 28 failed experiments alongside the 8
that worked.

The discipline the questions encode: the profitable surface on this platform is
mostly *above* the kernel — in dispatch gates, graph structure, and deleted
work — and [the failures page](the-failures.md) is what happens when effort goes
below it first.

Next: [Sparse-V](sparse-v.md)
