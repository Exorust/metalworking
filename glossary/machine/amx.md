# AMX

**AMX is Apple's undocumented CPU matrix coprocessor. Not part of the GPU at all,
but the other half of the Apple matmul story and the answer when small-matrix CPU
throughput seems impossible.**

CUDA equivalent: none. The closest analogy is Intel's AMX or ARM's SME (into which
Apple's AMX has been evolving), but the important point for a GPU person is
negative: when you benchmark Apple's Accelerate/BNNS small-matrix performance
and the numbers look too good for a CPU, you're not looking at the CPU's NEON
units. You're looking at a dedicated 32×32 multiply-accumulate grid hanging off the
CPU cluster, fed by its own X/Y register pools, executing an outer product per
instruction.

Why it appears in a GPU glossary:

- **Dispatch decisions.** For small matrices, the fastest processor on the chip is
  often AMX, not the GPU: kernel launch overhead plus
  [bandwidth](unified-memory.md) round trips swamp the GPU's advantage below a size
  threshold. Accelerate and BNNS, on the CPU side, can use it; MPS cannot, it
  encodes onto the GPU. If Accelerate beats your GPU kernel at 256×256, this is
  frequently why; if MPS does, look at the two GPU kernels.
- **It shares the memory pool.** [Unified memory](unified-memory.md) means
  CPU-side AMX work and GPU work compete for the same bandwidth; a background
  BNNS workload dents your GPU kernel's roofline.

![One die, three matrix engines: the CPU cluster with AMX, the GPU cores, and the Neural Engine, sharing the unified memory fabric, each reachable through a different API](../../diagrams/three-engines.svg)

*Where AMX sits. Three matrix engines share one bandwidth budget; Metal shaders
run only on the middle one, and the API rows at the bottom are the only ways to
reach each engine.*

Apple has never documented the instruction set. The definitive reference is
[corsix/amx](https://github.com/corsix/amx) (MIT): per-instruction docs with an
executable software emulator as the specification, plus per-generation deltas (M2
adds bf16; M3 adds new load/int modes; M4-era hardware moves toward standard SME).
Using AMX directly means issuing undocumented instructions, and Accelerate/BNNS is
the sanctioned interface. But reading corsix's docs is the only way to understand
the performance you observe through those libraries.

The [ANE war story](../war-stories/the-failures.md) is the same genre one chip
region over: the Neural Engine, also undocumented, reachable through Apple's
model frameworks (Core ML, and since macOS 27 Core AI for neural networks) or
private interfaces, at ~6.6 TFLOPS/watt. The GPU is one of *three* matrix
engines on this die, and Metal shaders run on only one of them; a Metal 4
[machine learning pass](https://developer.apple.com/documentation/metal/machine-learning-passes)
can hand a whole converted model to the Neural Engine, but the system decides
where it runs.

Next: [Neural accelerators](neural-accelerators.md)
