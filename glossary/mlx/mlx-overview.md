# MLX, an Overview

**MLX is Apple's open-source array framework for Apple Silicon: a NumPy-shaped
API with autograd, built natively on [unified memory](../machine/unified-memory.md),
whose GPU backend is the [steel kernel library](steel.md) this glossary reads.**

CUDA equivalent: the whole PyTorch + cuBLAS/cuDNN + bits-of-Triton stack, collapsed
into one codebase, which is why this glossary gives it a section rather than a
page. When you optimize ML on Apple Silicon you are usually optimizing *through*
MLX, and its architecture decides what's fast.

What a CUDA-stack person should recalibrate:

- **Arrays have no device.** `mx.array` lives in unified memory; there is no
  `.to("gpu")`, no host/device copies anywhere in user code. *Operations* have a
  location instead, a [stream on a device](lazy-evaluation.md), so CPU and GPU
  can genuinely cooperate on the same buffers (quantize on CPU while the GPU
  decodes, in the same address space).
- **Everything is lazy.** Operations build a graph; computation happens at
  [`mx.eval`](lazy-evaluation.md). This is not a compiler-ambition thing the way
  torch.compile is. It's primarily a [dispatch-batching](../metal/command-buffers.md)
  thing, and it's load-bearing for performance on this platform.
- **Function transforms, not autograd tape.** `mx.grad`, `mx.vmap`,
  [`mx.compile`](mx-compile.md) compose like JAX's transforms. Training exists and
  works; inference is where the ecosystem's energy is.
- **The kernels are readable.** Unlike cuBLAS, the fast path is open source MSL
  you can read: [steel](steel.md) for GEMM/attention,
  [quantized kernels](quantization.md) for LLM weights, with an
  [escape hatch for your own MSL](mx-fast.md).

The layer map, top to bottom (each layer has a page):

```
Python (mx.*)                        NumPy-shaped API, lazy
  graph + scheduler                  lazy-evaluation.md — batches work into command buffers
    primitives (eval_gpu)            how-an-op-becomes-a-kernel.md — C++ picks a kernel + tiles
      steel / quantized / fast MSL   steel.md, quantization.md, mx-fast.md — the actual kernels
        Metal                        ../metal/metal-the-api.md
```

Where the alternatives sit: **llama.cpp**'s Metal backend
([case study](../kernels/llamacpp-attention.md)) is the other production-grade
kernel body on the platform, C/C++ with no Python layer, organized around
quantized formats. **PyTorch MPS** runs and is improving: since 2.13 it ships hand-written
Metal kernels for hot paths like FlexAttention
([release notes](https://pytorch.org/blog/pytorch-2-13-release-blog/)) rather
than dispatching everything to [MPS](../metal/mps.md). It remains a port of a
CUDA-native design. MLX also now has a [CUDA
backend](https://ml-explore.github.io/mlx/build/html/install.html), making it
a two-backend framework rather than an Apple-only one. **MLX's Metal backend
is still where kernel-level work happens** if you want to touch the stack this
glossary teaches.

Next: [Lazy evaluation](lazy-evaluation.md)
