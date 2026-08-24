# MSL: Metal Shading Language

**MSL is the C++ dialect Metal kernels are written in: C++14 with GPU address
spaces, kernel attributes, and a SIMD standard library, compiled through
[AIR to GPU binary](compilation-pipeline.md).**

CUDA equivalent: CUDA C++. If you can read one you can read the other; here is the
accent, using a real kernel signature from the
[GEMM case study](../kernels/gemm-tiled.md):

```metal
kernel void __attribute__((max_total_threads_per_threadgroup(SW * SW * 32)))
matmul(
    constant uint  &n,
    constant uint  &k,
    const device float *A,
    device float       *C,
    ushort3 t_tg_pos       [[thread_position_in_threadgroup]],
    ushort3 tg_pos         [[threadgroup_position_in_grid]])
```
— [m5-gemm `sync_copy.metal:84-96`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L84-L96), abridged

Reading guide, CUDA → MSL:

| CUDA | MSL |
|---|---|
| `__global__ void f(...)` | `kernel void f(...)` |
| `__device__` function | plain function (or `METAL_FUNC` in MLX style) |
| pointer args | address-spaced: `device` (global memory), `constant` (uniform, cached), `threadgroup`, `thread` |
| `threadIdx` / `blockIdx` / `blockDim` | attribute-tagged parameters: `[[thread_position_in_threadgroup]]`, `[[threadgroup_position_in_grid]]`, `[[threads_per_threadgroup]]`, `[[thread_position_in_grid]]`, ... |
| `__shared__ float s[N];` | `threadgroup float s[N];` (in-kernel, static size) |
| `__launch_bounds__(n)` | `__attribute__((max_total_threads_per_threadgroup(n)))`, [worth 10× here](../machine/registers.md) |
| `#pragma unroll` | `#pragma clang loop unroll(full)` |
| `-D` compile defines | same, plus [function constants](function-constants.md) for post-compile specialization |

What MSL has that you'll actually use: real C++ **templates in device code**
(MLX's whole [steel library](../mlx/steel.md) is a template metaprogram),
`half`/`bfloat` as first-class types with `vec<T, N>` vectors, the
[`simdgroup_matrix`](simdgroup-matrix.md) types, and a `fast::` math namespace
(per-call fast math like [`fast::exp2`](../machine/special-paths.md), rather than a
whole-program `-use_fast_math` gamble).

What it lacks, coming from CUDA: no printf-debugging culture to speak of (it
exists; it's painful), no cooperative groups beyond the three built-in levels, no
dynamic parallelism, no dynamic threadgroup-memory sizing from inside the kernel
(the host sets it via `setThreadgroupMemoryLength`, or you declare statically), and
no inline PTX-equivalent. The one escape hatch,
[`__asm("air.*")` intrinsics, was closed by Metal 4](simdgroup-async-copy.md).

Hands-on is the fastest way in:
[Metal-Puzzles](https://github.com/abeleinin/Metal-Puzzles) is 14 progressively
harder MSL kernels (map → threadgroup memory → prefix sum → matmul) checked
against references from Python, and the recommended on-ramp before reading the
[case studies](../kernels/gemm-tiled.md).

Next: [Dispatch geometry](dispatch-geometry.md)
