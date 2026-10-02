# NVIDIA to Metal

**Every NVIDIA and CUDA term next to its Apple counterpart, on one page: the chip,
the kernel code, the toolchain, and the libraries. Each row says whether your
intuition transfers and links to the page that has the detail.**

Read this page first, then use it as the index for the rest of the glossary. The
other pages each explain one row.

## The levels

Two hierarchies exist at the same time. One is software: the threads that one
kernel launch creates. The other is hardware: the parts of the chip that run them.
Most confusion comes from mixing the two, so keep them in separate columns.

![Two hierarchies side by side. Software, top to bottom: a thread, a threadgroup of four simdgroups with 32 KB of threadgroup memory, and a grid of threadgroups over device memory. Hardware, top to bottom: an arithmetic unit, a GPU core with a 208 KB register file, 60 KB of threadgroup memory and an 8 KB L1 cache, and a GPU of 10 to 40 cores over unified memory shared with the CPU. Each software level runs on the hardware level beside it.](../../diagrams/architecture-levels.svg)

*Left: what one launch creates. Right: the hardware that runs it. Each row reads
across. The NVIDIA name for each box is under it.*

How the two columns connect:

- **A threadgroup runs on one core.** Its threads can share
  [threadgroup memory](threadgroup-memory.md) because that memory is inside the
  core.
- **A core holds several threadgroups at one time.** They divide its
  [registers](registers.md) and threadgroup memory between them, and the number
  that fit is the [occupancy](occupancy.md).
- **The core schedules [simdgroups](simdgroup.md), not single threads.** About 24
  simdgroups (about 768 threads) keep all its arithmetic units busy.
- **A grid is usually larger than the chip.** The threadgroups that do not fit
  wait, and each one starts when a core has room.
- **"Core" names a different level on each side.** An Apple
  [GPU core](gpu-core.md) is the size of an NVIDIA Streaming Multiprocessor. An
  NVIDIA "CUDA core" is one arithmetic unit inside it.

If you can change it in your code, it is software: the grid, the threadgroup size,
the number of simdgroups. If you can change it only with a different chip, it is
hardware.

## The chip

The last column uses three words:

- **Transfers**: the concept and the mechanism match. Reuse what you know.
- **Resized**: the same concept with different numbers, and the numbers change
  which kernel design wins.
- **Unlearn**: the CUDA habit is wrong here, because the feature is absent or its
  cost has the opposite sign.

| What it is | NVIDIA | Apple M-series | Verdict |
|---|---|---|---|
| Unit of compute | Streaming Multiprocessor (SM) | [GPU core](gpu-core.md) | Transfers |
| 32 threads that run in lockstep | warp | [simdgroup](simdgroup.md) | Transfers |
| Register file | ~256 KB per SM | [~208 KB per core](registers.md) | Resized: similar size, but here it is the main working memory |
| Scratch memory shared by a block | shared memory, up to 228 KB per SM, configurable | [threadgroup memory](threadgroup-memory.md), 32 KB per threadgroup, fixed | Resized: a staging buffer, not the place where tiles stay |
| Caches | 40 MB L2 (A100) | ~8 KB L1 data per core, a small L2, and an SLC tier | Resized: plan on explicit reuse, not cache locality |
| Main memory | HBM on the card at multiple TB/s; host RAM is across PCIe | [unified memory](unified-memory.md) shared with the CPU, ~153 GB/s (M5) to ~614 GB/s (M5 Max) | Unlearn: no transfers to manage, but bandwidth is the scarce resource |
| Threads needed to fill the ALUs | 2048 resident threads per SM | [~24 simdgroups (~768 threads) per core](occupancy.md) | Resized |
| Matrix unit | Tensor Core | none before M5, where [`simdgroup_matrix`](../metal/simdgroup-matrix.md) runs on the regular FP32 pipes; [neural accelerators](neural-accelerators.md) on M5 | Unlearn before M5 |
| Reason to use 16-bit floats | unlocks Tensor Core throughput | [shorter stalls and half the register pressure](f16.md) | Resized: same advice, different reason |
| Fast exponent | the SFU's `MUFU.EX2` | [`fast::exp2`](special-paths.md) | Transfers |
| Float atomics | `atomicAdd(float*)` is a cheap hardware instruction | [emulated](special-paths.md) | Unlearn |
| Copy engine that overlaps with compute | `cp.async` / TMA | [in the hardware but not exposed](../metal/simdgroup-async-copy.md); the Metal 4 compiler rejects it | Unlearn |
| Cost of a block-wide barrier | high enough that avoiding `__syncthreads()` is an optimization | [~2 cycles](../metal/synchronization.md) | Unlearn: barriers are nearly free |
| Largest part | datacenter GPU (H100) | laptop and desktop chip (M5 Max, 40 cores); there is no datacenter part | An H100 wins on raw FLOPs; Apple wins on memory capacity and efficiency |
| More than one GPU | NVLink inside one chassis | one device per machine; [Thunderbolt 5 RDMA](../mlx/distributed.md) between machines | Unlearn |

None of the Apple numbers come from Apple. The community measured them, chiefly in
[philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks).

## The kernel code

CUDA C++ and the Metal Shading Language ([MSL](../metal/msl.md)) are close enough
that you can read one if you can read the other.

| CUDA | Metal | Detail |
|---|---|---|
| grid, thread block, warp, thread | grid, threadgroup, simdgroup, thread | [Dispatch geometry](../metal/dispatch-geometry.md) |
| `__global__ void f(...)` | `kernel void f(...)` | [MSL](../metal/msl.md) |
| `threadIdx`, `blockIdx`, `blockDim` | attribute-tagged parameters such as `[[thread_position_in_threadgroup]]` | [MSL](../metal/msl.md) |
| `__shared__ float s[N];` | `threadgroup float s[N];` | [Threadgroup memory](threadgroup-memory.md) |
| `__launch_bounds__(n)` | `__attribute__((max_total_threads_per_threadgroup(n)))` | [Registers](registers.md) |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | [Synchronization](../metal/synchronization.md) |
| `__syncwarp()` | `simdgroup_barrier(mem_flags::mem_none)` | [Synchronization](../metal/synchronization.md) |
| `wmma` fragments, `mma.sync` | `simdgroup_matrix`, in 8×8 tiles | [simdgroup_matrix](../metal/simdgroup-matrix.md) |
| `cp.async` | no supported equivalent | [simdgroup_async_copy](../metal/simdgroup-async-copy.md) |
| TMA plus `wmma` descriptors | `MTLTensor` and MPP (Metal 4) | [MTLTensor and MPP](../metal/mtltensor-and-mpp.md) |
| one compiled kernel per configuration (template instantiation, NVRTC) | one kernel with function constants, bound when the pipeline is built | [Function constants](../metal/function-constants.md) |

## The toolchain and runtime

| CUDA | Metal | Detail |
|---|---|---|
| device and context | `MTLDevice`, usually one per chip | [Metal, the API](../metal/metal-the-api.md) |
| stream | `MTLCommandQueue` | [Command buffers](../metal/command-buffers.md) |
| kernel launch `<<<...>>>` | a dispatch encoded into a `MTLCommandBuffer`, then committed | [Command buffers](../metal/command-buffers.md) |
| `cudaMalloc`, then `cudaMemcpy` from the host | `MTLBuffer`; the CPU and GPU see the same bytes | [Unified memory](unified-memory.md) |
| nvcc | the `metal` frontend (`xcrun metal`), or compile at runtime | [Compilation pipeline](../metal/compilation-pipeline.md) |
| PTX | AIR, Apple's portable intermediate representation | [Compilation pipeline](../metal/compilation-pipeline.md) |
| fatbin | metallib | [Compilation pipeline](../metal/compilation-pipeline.md) |
| cubin, SASS | the GPU binary inside a pipeline state, always built on the user's machine | [Compilation pipeline](../metal/compilation-pipeline.md) |
| Nsight Compute | nothing comparable | [Profiling](../metal/profiling.md) |
| `cuobjdump`, `nvdisasm` | applegpu, a community disassembler | [Disassembly](../metal/disassembly.md) |

## The libraries

| NVIDIA stack | Apple stack | Detail |
|---|---|---|
| PyTorch, cuBLAS/cuDNN, and parts of Triton | MLX, one codebase | [MLX, an overview](../mlx/mlx-overview.md) |
| cuBLAS, cuDNN | MPS, which open kernels have matched or beaten | [MPS](../metal/mps.md) |
| CUTLASS | steel | [Steel](../mlx/steel.md) |
| cuDNN fused attention, the FlashAttention library | `mx.fast` | [mx.fast](../mlx/mx-fast.md) |
| Triton, for custom kernels written from Python | `mx.fast.metal_kernel` | [mx.fast](../mlx/mx-fast.md) |
| `torch.compile` | `mx.compile`, which does elementwise fusion and stops there | [mx.compile](../mlx/mx-compile.md) |
| PyTorch's asynchronous dispatch | lazy evaluation: nothing runs until `mx.eval` | [Lazy evaluation](../mlx/lazy-evaluation.md) |
| weight-only quantization kernels (TensorRT-LLM, AWQ, GPTQ) | MLX quantization | [Quantization](../mlx/quantization.md) |
| NCCL over NVLink | `mx.distributed` over Thunderbolt 5 RDMA | [Distributed](../mlx/distributed.md) |

## What to unlearn

The rows marked Unlearn and Resized become these habits:

1. **Keep the working set in registers, not in shared memory.** Threadgroup memory
   is 32 KB, so tiles pass through it and the results stay in
   [registers](registers.md). The failure is a spill, which ran
   [10× slower](../techniques/register-blocking.md) in the measured case.
2. **Use barriers freely.** A barrier costs ~2 cycles, so choose the algorithm with
   the cleaner staging pattern even if it [syncs more often](../metal/synchronization.md).
3. **Do not accumulate with float atomics.** They are
   [emulated](special-paths.md), and kernels that depend on them need a different
   structure.
4. **Do not plan on copy and compute overlap.** Without `cp.async`, tiles move by
   [cooperative loads](../techniques/cooperative-load.md), and
   [double buffering](../techniques/double-buffering.md) gets its overlap from
   instruction-level parallelism.
5. **Expect to be bandwidth bound.** More kernels are limited by memory bandwidth
   than CUDA experience suggests, so
   [arithmetic intensity](../techniques/arithmetic-intensity.md) decides most
   designs, and [fusion](../techniques/fusion-and-epilogues.md) and
   [quantization](../mlx/quantization.md) pay more here.
6. **Batch dispatches and sync rarely.** A launch is not a function call. Put many
   dispatches in one [command buffer](../metal/command-buffers.md) and wait for
   the GPU as rarely as possible.
7. **Build your own measurements.** There is no Nsight Compute, so you assemble the
   [roofline](../techniques/roofline.md) yourself and
   [change one thing at a time](../metal/profiling.md).

What transfers without change: warp intuition, the occupancy model,
[tiling](../techniques/tiling.md), and the algorithms themselves, such as
[online softmax](../techniques/online-softmax.md) and
[flash attention](../techniques/flash-attention.md).

Next: [GPU Core](gpu-core.md)
