# Stage 3: How production factors it

Time: one or two sittings of reading. Self-contained — every excerpt links back to the
pinned original ([ml-explore/mlx](https://github.com/ml-explore/mlx) @ `47bbfe8f`, MIT).

Stage 2 gave you a fast GEMM as one hand-written file with three `-D` constants.
Production can't ship that file. MLX has to serve every dtype (fp16, bf16, fp32,
complex), every transpose combination, matrices that don't divide evenly by the tile
size, batched and strided inputs, and a growing family of variants — split-K,
block-sparse, mixture-of-experts gather — from one codebase, without writing each
kernel by hand.

Its answer is **steel**, MLX's CUTLASS-style kernel library. The idea: take the kernel
you read in stage 2 and cut it at its natural joints — the cooperative load, the
register-tile multiply, the epilogue — into independent C++ templates that compose.
You already understand every piece; this stage is about seeing where each one went.

The map from stage 2 to steel:

| stage-2 concept | steel home |
|---|---|
| `load_tile<rows, cols, nthreads>` | `BlockLoader` (`steel/gemm/loader.h`) |
| accumulators + `simdgroup_multiply_tile` | `BlockMMA` (`steel/gemm/mma.h`) |
| the `α·AB + β·C` ending | `Epilogue` transforms (`steel/gemm/transforms.h`) |
| `SW`, `SIMD_TILE`, `TILE_K` | template params `BM, BN, BK, WM, WN` |
| the `matmul` kernel function | `steel_gemm_fused.h`, which just wires the above together |

## 1. BlockLoader: the cooperative load, generalized

Here is the entire interface — the template parameters *are* the design:

```cpp
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoader {
```
— [`steel/gemm/loader.h:14-25`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L14-L25)

Read the defaulted parameters closely — they're doing the arithmetic you did by hand
in stage 2. `n_reads` = tile elements ÷ threadgroup size = elements per thread (stage
2's "each thread walks 2 float4s"). From it, `TCOLS`/`TROWS` derive how the
threadgroup's threads tile across the slab, *at compile time*, for any tile shape and
any threadgroup size. What m5-gemm hard-coded for one shape, this derives for all
shapes.

The hot-path load is stage 2's coalesced copy, now with vectorization expressed
through a type instead of a `float4` cast:

```cpp
  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };
```
```cpp
  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }
```
— [`steel/gemm/loader.h:42-44, 73-80`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L73-L80)

One `ReadVector` copy per row: the aligned struct makes the compiler emit the widest
load the alignment allows. And note what's new versus stage 2: `load_unsafe` has a
sibling, **`load_safe(short2 src_tile_dim)`**
([lines 83-128](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L83-L128)),
which bounds-checks and zero-fills when the tile hangs off the matrix edge. m5-gemm
simply required sizes divisible by 64. Production doesn't get to require that — and
you'll see below how the kernel avoids paying for the checks when it doesn't have to.

Last detail, easy to miss, load-bearing for the whole design:

```cpp
  /* Iteration helper */
  METAL_FUNC void next() thread {
    src += tile_stride;
  }
```
— [`steel/gemm/loader.h:130-133`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L130-L133)

The loader owns its own pointer arithmetic. The kernel's K-loop never computes an
address; it just calls `next()`. That's what makes the loader swappable — a quantized
loader or a transposed loader presents the same three verbs (`load`, `next`, done) and
the kernel body doesn't change.

## 2. BlockMMA: the register tile, generalized

```cpp
struct BlockMMA {
  // MMAFrag size
  STEEL_CONST short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  ...
  // Warp tile size along M
  STEEL_CONST short TM = BM / (kFragSize * WM);
  // Warp tile size along N
  STEEL_CONST short TN = BN / (kFragSize * WN);
  ...
  // Simdgroup matrices
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;
```
— [`steel/gemm/mma.h:453-483`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/mma.h#L453-L483), abridged

Decode the names against stage 2: `kFragSize=8` is the hardware's 8×8
`simdgroup_matrix`. `WM × WN` is how the threadgroup's simdgroups are arranged (stage
2's `SW × SW`). `TM × TN` is each simdgroup's grid of accumulator fragments (stage 2's
`SIMD_TILE × SIMD_TILE` array of `acc[i][j]`). `Ctile` — the `TM × TN` fragment tile
living in registers for the whole kernel — is exactly stage 2's accumulator array,
with the register-pressure implications you already know.

The multiply is stage 2's K-loop inner body, one `kFragSize` slice at a time:

```cpp
  /* (BM, BK) X (BK, BN) multiply accumulate function */
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) thread {
    // Adjust for simdgroup and thread location
    As += As_offset;
    Bs += Bs_offset;

    // Iterate over BK in blocks of kFragSize
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);
      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);
      simdgroup_barrier(mem_flags::mem_none);
      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Ctile, Atile, Btile, Ctile);

      // Progress to next simdgroup tile
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }
```
— [`steel/gemm/mma.h:512-537`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/mma.h#L512-L537)

Same shape as `simdgroup_multiply_tile` in stage 2: load A fragment, load B fragment,
multiply-accumulate, step K. The differences are generality (transpose handled by the
`A_str_m/A_str_k` stride constants — swap the strides, not the code) and the
`simdgroup_barrier(mem_none)` calls, which are scheduling hints keeping the simdgroup's
loads together rather than full threadgroup syncs.

## 3. Epilogues: the ending as a plug-in

Stage 2's kernel ended with `α·AB + β·C` fused into the store. Steel makes the ending
a template parameter. The entire vocabulary:

```cpp
template <typename OutT, typename InT>
struct TransformAdd {
  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;
  METAL_FUNC OutT apply(InT x, OutT c) const thread {
    return static_cast<OutT>(
        x * static_cast<InT>(alpha) + (static_cast<OutT>(beta) * c));
  }
};
```
— [`steel/gemm/transforms.h:25-54`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/transforms.h#L25-L54), abridged

`BlockMMA::store_result` applies the epilogue to each register-resident accumulator
element on its way out
([`mma.h:540-551`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/mma.h#L540-L551)).
This is stage 0's lesson 1 as an API: anything expressible as an epilogue — bias add,
activation, scaling — costs zero extra memory passes, because it happens while the
result is still in registers. The alternative (a second kernel) would re-read and
re-write all of C through DRAM on a bandwidth-bound machine.

## 4. The kernel is now just wiring — plus one new trick

With loading, math, and endings factored out, the actual GEMM kernel is short on
ideas and long on dispatch. Its header shows the one genuinely new mechanism:

```cpp
constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

// clang-format off
template <
    typename T,
    int BM, int BN, int BK,
    int WM, int WN,
    bool transpose_a, bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm(
```
— [`steel/gemm/kernels/steel_gemm_fused.h:9-29`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L9-L29), reformatted

(Notice `max_total_threads_per_threadgroup` — stage 2's single biggest win, here as
standard practice on every steel kernel.)

Two specialization mechanisms are in play, and the distinction matters for stage 4:

- **Template parameters** (`T, BM, BN, ...`) are fixed when MLX's build generates the
  kernel-instantiation list. Each combination is a separate kernel in the metallib.
- **Function constants** (`align_M`, `has_batch`, ...) are bound at *pipeline creation*,
  when the host knows the actual problem. The Metal compiler then finalizes the kernel
  with these as literal constants — branches on them are resolved and dead code is
  eliminated **before the GPU ever runs it**. Specialization without a combinatorial
  explosion of compiled kernels.

Watch what `align_M`/`align_N` buy in the body:

```cpp
  // MNK aligned loop
  if (align_M && align_N) {
    // Do gemm
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Load elements into threadgroup
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }
```
— [`steel/gemm/kernels/steel_gemm_fused.h:172-188`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L172-L188)

That inner loop *is* stage 2's K-loop — load, barrier, mma, advance — written once
against the component interfaces. When the host proves the matrix divides evenly by
the tile (`align_M && align_N` true), the compiled kernel contains only this fast
path: `load_unsafe`, no bounds checks anywhere. The unaligned branches below it
([lines 209-345](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L209-L345))
handle every ragged-edge combination via `load_safe`/`store_result_safe` — and vanish
from the aligned pipeline entirely. Edge-case handling that costs nothing unless
you're on the edge: that's the trick m5-gemm never needed and production can't live
without.

## 5. The shape space one template family covers

Skim the sibling kernels in
[`steel/gemm/kernels/`](https://github.com/ml-explore/mlx/tree/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels)
just for their names: `steel_gemm_splitk.h` (K huge relative to M·N — split the
reduction across threadgroups, combine after), `steel_gemm_masked.h` (block-sparse),
`steel_gemm_gather.h` (mixture-of-experts gather-GEMM), `steel_gemm_segmented.h`.
Every one reuses `BlockLoader` and `BlockMMA` and swaps the wiring. That reuse is the
entire argument for the decomposition you just read.

Two neighbors worth knowing exist, both fought over in stage 6's war stories:
`quantized.h` (the group-wise quantized matmul/QMV kernels serving every 4-bit LLM on
a Mac) and the `*_nax` variants (`gemm_nax.h` etc.) — the Metal-4/M5-era tensor-op
path. Diffing `gemm_nax.h` against its plain sibling shows exactly what the new
hardware instruction set changes; nothing else in the structure moves.

> Orientation warning for stage 4: there are *two* steel subsystems, `steel/gemm/` and
> `steel/attn/`, each with its own `mma.h`/`loader.h` and different fragment layouts.
> You just read `gemm/`. Attention gets its own fork because its middle matrix (the
> attention scores) must stay in registers through a softmax — stage 4 shows why that
> breaks the clean GEMM factorization.

## Optional deep end: the compiler school

Luminal generates kernels instead of templating them:
[Compiling fast GPU kernels](https://docs.luminalai.com/blog/gpu) — elementwise fusion
via e-graph rewrites, one command buffer for all of Llama 3 8B, search-based
compilation that rediscovers flash attention. The MSL lives in one Rust file,
`code/luminal/crates/luminal_metal/src/kernel/ops.rs` (3468 lines): each op emits a
`kernel void` string with shapes and dtypes substituted at compile time. Interesting
as the third answer to "how do you avoid hand-writing every kernel" — templates
(steel), string codegen (luminal, and stage 4's metal-flash-attention), or a fixed
specialization table (stage 4's llama.cpp).

## Done when

- You can name what each of `BM, BN, BK, WM, WN` controls and match each to its
  stage-2 ancestor (`SW`, `SIMD_TILE`, `TILE_K`).
- You can explain the difference between a template parameter and a function constant:
  when each is bound, and what the `align_M/align_N` constants eliminate from the
  compiled kernel.
- You can say what an epilogue is, why `TransformAxpby` running in `store_result`
  costs no extra memory traffic, and which stage-0 lesson that is.
- You can explain why `attn/` needs its own `mma.h` instead of reusing `gemm/`'s
  (softmax between the two matmuls; scores must stay in registers).

Next: [Stage 4: Flash Attention, three ways](stage-4-attention.md)
