# AMX

**AMX is Apple's undocumented CPU matrix coprocessor. Not part of the GPU at all,
but the other half of the Apple matmul story and the answer when small-matrix CPU
throughput seems impossible.**

CUDA equivalent: none. The closest analogy is Intel's AMX or ARM's SME (into which
Apple's AMX has been evolving), but the important point for a GPU person is
negative: when you benchmark Apple's Accelerate/BNNS/MPS small-matrix performance
and the numbers look too good for a CPU, you're not looking at the CPU's NEON
units. You're looking at a dedicated 32×32 multiply-accumulate grid hanging off the
CPU cluster, fed by its own X/Y register pools, executing an outer product per
instruction.

Why it appears in a GPU glossary:

- **Dispatch decisions.** For small matrices, the fastest processor on the chip is
  often AMX, not the GPU: kernel launch overhead plus
  [bandwidth](unified-memory.md) round trips swamp the GPU's advantage below a size
  threshold. Apple's own libraries route accordingly. If your profiling shows MPS
  beating your GPU kernel at 256×256, this is frequently why.
- **It shares the memory pool.** [Unified memory](unified-memory.md) means
  CPU-side AMX work and GPU work compete for the same bandwidth; a background
  BNNS workload dents your GPU kernel's roofline.

Apple has never documented the instruction set. The definitive reference is
[corsix/amx](https://github.com/corsix/amx) (MIT): per-instruction docs with an
executable software emulator as the specification, plus per-generation deltas (M2
adds bf16; M3 adds new load/int modes; M4-era hardware moves toward standard SME).
Using AMX directly means issuing undocumented instructions, and Accelerate/BNNS is
the sanctioned interface. But reading corsix's docs is the only way to understand
the performance you observe through those libraries.

The [ANE war story](../war-stories/the-failures.md) is the same genre one chip
region over: the Neural Engine, also undocumented, also reachable only through
private interfaces, at ~6.6 TFLOPS/watt. The GPU is one of *three* matrix engines
on this die, and Metal only talks to one of them.

Next section: [Metal, the stack](../metal/metal-the-api.md)
