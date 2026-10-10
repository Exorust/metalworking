# How an op becomes a kernel

## Where we are

Chapter 16 left the matrix multiply as a graph node that the scheduler hands to
its primitive at evaluation time. Between that node and a `simdgroup_matrix`
instruction sits one C++ hop: the primitive's `eval_gpu`, which picks a kernel
family, picks tile sizes, binds function constants, and encodes the dispatch.
In CUDA this is the dispatcher layer between PyTorch and cuBLAS's heuristics,
except that here the code is open, small and readable. Once you know where these
decisions live, you can turn "my model is slow" into "my shapes fall off the fast
path". The second of the three questions of Chapter 24, unlock an existing fast
path, is answered by reading exactly these files.

## The chain, for a matmul

The lazy graph hands `Matmul` to the scheduler. `Matmul::eval_gpu`, in
`mlx/backend/metal/matmul.cpp`, first normalizes each operand: if neither stride
is 1 it encodes a copy to a contiguous layout before the multiply, and it
collapses batch dimensions where it can. Then it routes, in this order.

1. `gemv_wide`, for a 16-bit multiply with A untransposed and B transposed, 2
   to 15 rows and more than one column, on M3-class chips and later, when K is
   a multiple of 4 and the operands' offsets and strides pass its alignment
   checks.
2. `gemv`, when M or N is 1.
3. `steel_matmul_axpby`, for everything else that is not empty. Inside it, the
   first decision is the hardware: `use_nax` is true when `is_nax_available()`
   holds, which on a Mac needs a generation-17 or later GPU, macOS 26.2 or later
   and a build without `MLX_METAL_NO_NAX`, when the type is not complex, and
   when either the type is not 32-bit float or TF32 is enabled, which
   `MLX_ENABLE_TF32` turns on by default. If `use_nax` is true the multiply
   returns through the NAX fork of Chapter 29, with its own split-K branch for
   large K, and never reaches the macro below. If it is false, an unbatched
   multiply whose output is small against K (at most 2,048 tiles of 16 by 16 on
   a Max or Ultra, K at least 128 and at least max(M, N)) takes the plain
   split-K kernel. Only then does the tile macro pick the plain steel kernel's
   shape.

```cpp
#define GEMM_TPARAM_MACRO(devc)                                           \
  if (devc == 'g' || devc == 'p') { /* Small device */                    \
    ...
  } else if (devc == 'd') { /* Large device */                            \
    if ((size_t)batch_size_out * M * N >= 1ul << 20) { /* large matmul */ \
      if (out.dtype() != float32) { /* half and bfloat */                 \
        if (2 * std::max(M, N) > K) { /* Reasonable K */                  \
          bm = 64;                                                        \
          bn = 64;                                                        \
          bk = 16;                                                        \
          wm = 1;                                                         \
          wn = 2;                                                         \
        } else if (!transpose_a && transpose_b) { /* nt with large k */   \
          bm = 64;                                                        \
          bn = 32;                                                        \
          bk = 32;                                                        \
          wm = 2;                                                         \
          wn = 2;                                                         \
```
Source: [`mlx/backend/metal/matmul.cpp:89-124`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/matmul.cpp#L89-L124), abridged

Read what that macro is: a hand-tuned lookup from the chip class to a tile
shape, `bm`, `bn`, `bk`, `wm`, `wn`, which on the largest class also looks at
the size, the data type and the transpose pattern. Those are the `BM`, `BN`,
`BK` of Chapter 11 and the simdgroup arrangement of Chapter 12, chosen per shape
instead of fixed by `-D` defines. It does the job of cuBLAS's secret heuristics,
but you can search it. The device class letter is the last character of the
Metal architecture name, and `device.cpp` spells out the mapping: `p` for phone,
`g` for base and Pro chips, `s` for Max, `d` for Ultra. The macro's small-device
branch is `g` and `p`; its large-device branch is `d`; everything else, which
includes every Max, falls into a final `else` the file labels "Medium device"
and gets `bm` = `bn` = 64, `bk` = 16, `wm` = `wn` = 2, the same values the
function sets as defaults before the macro. That is a 64 by 64 tile, 16 deep,
four simdgroups, the hand-written kernel's dimensions.

The ownership differs, though. Steel's `BlockMMA` places simdgroup fragments on
a lattice: fragment rows are spaced 8 × `wm` apart and fragment columns 8 × `wn`
apart, with each simdgroup's origin at 8 × (id / `wn`), 8 × (id % `wn`). With
`wm` = `wn` = 2 each simdgroup owns 32 by 32 elements interleaved with the
other three on a 16-element stride, not a contiguous quarter as in Part 4.
Chapter 18 reads that code.

On an Ultra, the first row shown gives 16-bit matrices of a reasonable shape,
where 2 × max(M, N) exceeds K, a 64 by 64 tile, 16 deep, with 1 by 2
simdgroups. The second row covers A untransposed and B transposed when 2 × max(M, N) is at
most K; the other transpose patterns at that K take a 32 by 64 tile that the
excerpt omits. A 32-bit large matmul on an Ultra takes the defaults, which the file
comments mark with "float takes default".

Then the kernel name is assembled as a string in two stages. The base name,
`steel_gemm_fused_...bm64_bn64...` and so on, names one template instantiation,
and the pinned `steel_gemm_fused.metal` precompiles exactly six tile shapes per
type into the shipped library; `MLX_METAL_JIT` is off by default, so a shape
outside those six is not compiled at run time. The hash name appends the six
function constants of Chapter 8, `has_batch`, `use_out_source`, `do_axpby`,
`align_M`, `align_N` and `align_K`, and keys the pipeline state, which the
backend compiles on the first miss (Chapter 6) and the cache serves after.
Matrix sizes are run-time arguments, so two shapes that pick the same template
and the same alignment flags share one pipeline.
The dispatch is then encoded into the shared command encoder of Chapter 16.

![The MLX matmul dispatch tree: shape checks route to gemv or quantized families, otherwise the tile-selection macro picks steel parameters, the kernel name is assembled, and the pipeline is fetched or JIT built](../diagrams/dispatch-tree.svg)

*The route a matmul takes. The amber branches are where shapes silently leave
the steel fast path; the green spine is the path Chapter 18 reads.*

## The branches before steel

Several branches come before the macro, and a shape that falls into the wrong
one runs slower without any warning.

- **Matrix-vector shapes** route to `gemv` when M or N is 1, and 16-bit
  transposed-B multiplies with up to 15 rows route to `gemv_wide` on M3-class
  chips and later, because a few rows cannot fill a tile (Chapter 22).
- **Small outputs with a large shared dimension** take a split-K kernel: the
  plain one under the conditions above when NAX is off, and a NAX one when K is
  at least three times max(M, N), or more than twice it below 1,024.
- **The hardware gate** sends every eligible multiply on an M5-class GPU to the
  NAX fork, including 32-bit multiplies unless TF32 is disabled.
- **Quantized weights** are a separate primitive with a different kernel family
  (Chapter 23), whose fast path is shape-sensitive in its own ways.
- **`mx.fast.scaled_dot_product_attention`** is likewise its own primitive, with
  its own gate deciding between the fused kernel and a fallback graph (Chapter
  21).

The measured example of the trap is the MTPLX case of Chapter 24: 2.24 times
end-to-end from about 10 lines, which came from noticing that multi-token decode,
3 to 6 rows at a time, stalled the quantized matrix-vector kernel of Chapter 23,
which is tuned for one row, while the quantized tiled kernel is tuned for many.
The kernel existed, but its tuning grid had a hole. The same reading applies to
every family: find the branch your shape takes, then the condition that sent it
there.

## What this means for the kernel writer

Because the dispatcher is readable, three habits follow.

First, before writing anything, read the branch your shape takes. The question
"why is this slow" often has the answer "because it took the branch for a
different shape", and the fix is a change to the dispatch rather than a new kernel.

Second, the tile table is where production code applies the "no single fastest
kernel" lesson of Chapter 14. The table encodes the tile picked per chip
class, and a new generation changes the route not by editing it but through the
hardware gate above it: an M1 Max reads the "Medium device" `else`, and an
eligible M5 Max multiply has already returned through the NAX fork before the
macro runs.

Third, the pipeline cache is keyed by the hash name, so a model that meets many
distinct template and constant combinations pays a pipeline build for each the
first time, and MLX keeps them until the library that owns them is cleared.

## Where the matmul stands

For 4096 by 4096 in 32-bit on an M5 Max with the pinned MLX, on macOS 26.2 or
later and a default build, the hardware gate fires: `is_nax_available()` is true
and TF32 is on by default, so the multiply goes to `steel_gemm_fused_nax` with a
64 by 128 tile, 256 deep, eight simdgroups, the fork Chapter 29 reads; note that
TF32 changes the 32-bit multiply's numerics. On a Max before M5, or on an M5
Max with `MLX_ENABLE_TF32=0`, it takes the plain path this part follows: the
Max class lands in the "Medium device" `else`, `bm` = `bn` = 64, `bk` = 16,
`wm` = `wn` = 2, the base name `steel_gemm_fused_...` names that precompiled
instantiation, the `align_M`, `align_N` and `align_K` constants bind to true
because 4096 % 64 = 0 and 4096 % 16 = 0, and the aligned pipeline is
dispatched. A base or Pro chip, class `g`, would read the small-device branch
instead. That
kernel is the subject of the next chapter.

## Check yourself

1. A model runs a 4096 by 4096 matrix multiply in 16-bit with K = 64 on an M2
   Ultra. Which branch of the excerpt does it take, and which tile shape does it
   get? What changes on an M5 Max?
2. A speculative decoder sends 4 query rows at a time against quantized weights
   and runs slower per row than single-row decode. Which part of the dispatch
   chain does the chapter say to read first, and why?
3. Why does MLX assemble the kernel name as a string rather than calling one
   kernel, and what are the two keys?

### Answers

1. On the Ultra, device class `d` and a generation-14 GPU, so NAX is off: the
   large-device, large-matmul branch applies because 4096 × 4096 exceeds 2^20,
   the output is not 32-bit, and `2 × max(M, N) > K` is 8192 > 64, so the
   "reasonable K" row, `bm` = 64, `bn` = 64, `bk` = 16, `wm` = 1, `wn` = 2. On
   the M5 Max, with the OS and build conditions met, the hardware gate fires
   first and the 16-bit multiply goes to the NAX fork with a 64 by 128 by 256
   tile, `wm` = 2, `wn` = 4; the macro never runs, and K = 64 leaves a tail
   there, since 64 % 256 is not 0.
2. The quantized family's own dispatch (Chapter 23), because its matrix-vector
   kernel is tuned for one row and its tiled kernel for many, and 4 rows falls
   between them; the MTPLX case found exactly that hole and fixed it in about 10
   lines of the matrix-vector kernel.
3. Because tile sizes are template parameters (Chapter 8), so each tile shape
   is a separate compiled function with its own host name, and MLX chooses to
   select it by assembling that name: the base name keys the library function,
   and the hash name, which appends the six function constants, keys the
   pipeline state.

The dispatch has chosen `steel_gemm_fused` on the plain path, and the next
chapter reads it as the hand-written kernel of Part 4 rebuilt from three
components.

Next: [Steel: the GEMM as components](18-steel-the-gemm-as-components.md)
