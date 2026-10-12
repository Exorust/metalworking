# Three matrix engines on one die

## Where we are

Every kernel you read in Parts 4 to 6 uses `simdgroup_matrix`, which runs
on the GPU's regular arithmetic pipes, and until the M5 generation that is all
the GPU had: Chapter 12 said `simdgroup_matrix` is the API of a Tensor Core on
hardware that had none. The benchmarks ran on an M5 Max, and Chapter 17 showed
MLX's own dispatcher already taking the other route there; what you read was the
portable path. The new generation gave the GPU a dedicated matrix unit, and
reaching it takes a different programming model. Before the new unit, step one level up to the three matrix engines that share
one Apple die, because two of them affect GPU work whether or not you ever
program them, and the third is the new one.

## The die has three matrix engines

![One die, three matrix engines: the CPU cluster with AMX, the GPU cores, and the Neural Engine, sharing the unified memory fabric, each reachable through a different API](../diagrams/three-engines.svg)

*Where the engines sit. Three share one bandwidth budget; Metal talks to the
middle one, and the API rows at the bottom are the only ways to reach each.*

**AMX, on the CPU.** When Accelerate beats your GPU kernel on a small matrix,
AMX is often the reason. It is Apple's undocumented CPU matrix coprocessor,
not part of the GPU, but it is the other half of Apple's matrix-multiply
performance, and it explains small-matrix CPU throughput that NEON alone could
not deliver. There is
no CUDA counterpart; the closest analogies are Intel's AMX and ARM's SME, toward
which Apple's AMX has been evolving. The point for you is negative:
when you benchmark Apple's Accelerate, BNNS or MPS small-matrix performance and
the numbers look too good for a CPU, you are not looking at the CPU's NEON units.
You are looking at a dedicated multiply-accumulate grid hanging off the CPU
cluster, fed by its own register pools, executing an outer product per
instruction.

Apple has never documented the instruction set. The definitive
reference is corsix/amx (MIT licensed), which has per-instruction documentation, an executable
emulator that is itself the specification, and per-generation deltas: M2 adds bf16, M3 adds
new load and integer modes, M4-era hardware moves toward standard SME. Using AMX
directly means issuing undocumented instructions. Accelerate and BNNS are the
sanctioned interface, and corsix's documentation is where you find out what
those libraries can be doing underneath the numbers you measure.

AMX matters to GPU work in two ways. The first is dispatch: for small matrices, the
fastest processor on the chip is often AMX, not the GPU. Below a size threshold,
kernel launch overhead plus the round trips of Chapter 7 swamp the GPU's
advantage, and Accelerate and BNNS, which run on the CPU side, can use AMX.
Metal Performance Shaders cannot: MPS encodes its matrix multiply into a command
buffer and runs it on the GPU, which is how Chapter 9 timed it with GPU
timestamps. So if Accelerate beats your kernel at 256 by 256, this is frequently
why; if MPS does, the explanation is in the GPU kernel and its dispatch, not in
AMX.

The second is the shared memory pool: unified memory means CPU-side AMX
work and GPU work compete for the same bandwidth, so a background BNNS workload
can cut the bandwidth a bandwidth-bound GPU kernel achieves.

**The Neural Engine.** The third engine is also undocumented, and you can reach it
only through Apple's model frameworks or private interfaces. It delivers about 6.6
TFLOPS per watt against about 1 for the GPU (Chapter 25). Metal does not talk to it.

**The GPU's own matrix unit, from M5.** The M5 generation adds a dedicated matrix
unit to every GPU core: the neural accelerators, NAX in MLX's source, reached
through Metal 4 tensor operations rather than through `simdgroup_matrix`. This is
the first real Tensor Core counterpart on the GPU.

## What M5 changes, and what it does not

What did M5 add under your kernels? The `simdgroup_matrix` kernels of the
earlier chapters execute on the regular 32-bit pipes, and the reverse-engineered Rigel work
confirmed no dedicated matrix datapath as late as M4. On M5 that changed, and
Apple's own numbers for MLX on M5 say where: about 3.97 times faster time to
first token and 1.19 to 1.27 times faster decode than M4, end-to-end numbers
that fold in everything else that changed between the chips. The prefill gain
is where a matrix unit should show, in the compute-bound territory of
Chapter 3. Decode stays bandwidth bound, so it moves with the memory spec
instead (Chapter 22).

So what should you carry over from CUDA, and what not? Start with the fragment
shape, which moves from 8 by 8 to 16 by 16. MLX's NAX steel is built on a 16 by
16 fragment held across the 32 threads of a simdgroup.

```cpp
struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;
```
Source: [MLX `steel/gemm/nax.h:27-31`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/nax.h#L27-L31)

The file includes `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`
at the top: the new hardware arrives through a library, not through new
intrinsics. Each thread holds 16 × 16 / 32 = 8 elements of a fragment, against
2 for an 8 by 8 fragment. The fragment is MLX's storage unit; the operation it
hands to the library is wider. The same file builds a
`matmul2d_descriptor(16, 32, 16, ...)`, one A fragment against two B fragments
side by side, and the tile loops Chapter 29 reads combine those operations.

The formats are block-scaled floats. The `mxfp4`, `nvfp4` and `mxfp8` modes of
Chapter 23 arrived with this unit and have NAX kernels, but in the pinned MLX
those kernels dequantize the codes to 16-bit values before loading tiles,
Chapter 23, so that code does not show the hardware taking the packed formats
directly. The "fp8 is emulated" caveat of Chapter 4 is an M4-era fact.

And it is not programmed like a Tensor Core. There is no `mma.sync` to descend
to. You describe a matrix multiply with a descriptor and cooperative tensors,
and the compiler and the hardware choose the datapath. Chapter 29 covers the
model and reads real code.

How ready is the software? Support is early. MLX ships a parallel steel fork for it (`gemm/nax.h`,
`attn/nax.h`, `quantized_nax` and `fp_quantized_nax`), llama.cpp has a tensor-API
matrix-multiply path, and the tuning is young. A disproportionate share of recent MLX work targets NAX: it
raised batch limits for M5-class GPUs, optimized NVFP4 matrix-vector kernels,
and added per-expert tile picks. The path is still
changing quickly. `simdgroup_matrix` remains the portable primitive. It runs on
every M-series chip and most shipped kernels still use it, which is why this
book teaches both.

## Where the matmul stands

On a 40-core M5 Max, the running example has a second route: the tiled matrix
multiply of Part 4 still runs on the regular pipes at the 13.5 TFLOPS Chapter 10
measured in 32-bit, and the same multiply described to the neural accelerators
runs on a dedicated unit with 16 by 16 fragments. If you compare the two with
32-bit inputs, state the precision and check the error: the second route
multiplies in TF32, which changes the numerics, Chapter 17. The next
chapter reads the code for the second route.

## Review the content

1. Accelerate beats your hand-written GPU matrix multiply at 256 by 256 but
   loses at 4096 by 4096. What does the chapter say is probably happening, and
   would the same explanation hold if it were MPS winning?
2. Apple reports M5 prefill about 4 times faster than M4 and decode about 1.2
   times faster. Why the difference?
3. How many elements of a 16 by 16 fragment does each thread of a simdgroup
   hold, and how does that compare with the 8 by 8 fragment of Chapter 12?

### Answers

1. At 256 by 256, kernel launch overhead and the round trips of Chapter 7 swamp
   the GPU's advantage, and the fastest processor for that size is often AMX on
   the CPU, which Accelerate can use; at large sizes the GPU's throughput wins,
   and the chapter gives no measured crossover, so find yours by measuring. If
   MPS were winning, no: MPS runs on the GPU, so the explanation would be in the
   two GPU kernels, not in AMX.
2. Prefill is compute bound, so a new matrix unit raises its ceiling; decode at
   batch one on a dense model is bandwidth bound, so it moves mostly with the
   memory bandwidth, which changed by a smaller factor. Batching,
   mixture-of-experts routing and encoding overhead can shift that balance,
   Chapters 7 and 22.
3. 16 × 16 / 32 = 8 elements per thread, against 8 × 8 / 32 = 2 for the smaller
   fragment, and 8 / 2 = 4 times the elements per fragment at the same element
   type. That is payload per fragment, not more registers per output tile: four
   8 by 8 fragments cover the same 16 by 16 patch with the same element count.

To take the accelerator route you have to describe the multiply with descriptors
and cooperative tensors, and the last chapter reads that model and a kernel
written for it, ending where the book began, with a matrix multiply.

Next: [Tensor ops and the NAX GEMM](29-tensor-ops-and-the-nax-gemm.md)
