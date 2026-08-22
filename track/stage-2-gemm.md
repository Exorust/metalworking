# Stage 2: The GEMM ladder

Time: two or three sittings of reading; one more if you run the benchmarks (optional,
needs Apple Silicon). All code below is quoted in full context — you don't need to open
the source files, but every excerpt links back to its exact line in the pinned original.

Every technique that matters in GPU performance work — tiling, threadgroup staging,
register accumulation, latency hiding — shows up in matrix multiply in its purest form.
This stage walks one real, current, fast kernel line by line, then its double-buffered
sibling, then the ghost of the technique that preceded them both.

The code is from [yaroslavvb/m5-gemm](https://github.com/yaroslavvb/m5-gemm)
(BSD-3-Clause, a Metal-4 port of Zeke Medley's
[metal-matmul](https://github.com/0xekez/metal-matmul) and his essay
[Fast Multidimensional Matrix Multiplication on Apple GPU](https://percisely.xyz/gemm)).
The essay is worth reading in full, but this page is self-contained.

![Three Metal GEMM kernels benchmarked across matrix sizes, showing the fastest one changes with the problem size](../gemm-ladder.gif)

The punchline up front: **there is no single fastest kernel.** Double buffering wins at
1024, the simpler version wins at 4096, and Apple's MPS takes it back at 8192. By the
end of this page you'll know why the winner flips. (Numbers are m5-gemm's, measured on
an M5 Max — not measured here.)

## 1. Why naive matmul is slow: you pay for memory, not math

C = A×B at size n does 2n³ floating-point operations on 3n² numbers. That ratio —
FLOPs per byte touched, called **arithmetic intensity** — is the whole game. Stage 0's
lesson was that Apple Silicon has enormous compute relative to its memory bandwidth, so
a kernel that re-reads its inputs from DRAM for every output element (which is what the
naive triple loop does: every C[i][j] re-reads a full row of A and column of B) is
bandwidth-starved long before the ALUs break a sweat.

The fix, on every GPU ever made, is **reuse through the memory hierarchy**: pull a
block of A and B on-chip once, and do as much math as possible against it before
letting it go. On Apple Silicon the hierarchy you can program is:

- **device memory** — DRAM, hundreds of GB/s, shared with the CPU (unified memory)
- **threadgroup memory** — 32 KB of fast on-chip scratch, shared by one threadgroup
- **registers** — the ~208 KB-per-core register file; the fastest and, as you'll see,
  the scarcest resource

A fast GEMM is three nested levels of "copy a tile closer, then multiply":

```
grid            → each threadgroup owns a BM×BN tile of C
  threadgroup   → stages BM×BK of A and BK×BN of B in threadgroup memory
    simdgroup   → each simdgroup owns a sub-tile of C, held in registers
      hardware  → simdgroup_multiply_accumulate does an 8×8 matmul per instruction
```

`simdgroup_float8x8` is Apple's "tensor core" primitive: a special register type that
holds an 8×8 fp32 matrix spread across the 32 threads of a simdgroup, with a
single-instruction multiply-accumulate. You never index into it; you `simdgroup_load`
it from memory, accumulate into it, and `simdgroup_store` it back.

## 2. The kernel, top to bottom

This is `sync_copy.metal` — the current fastest hand-written kernel in this lineage,
13.5 TFLOPS on an M5 Max vs 11.7 for Apple's MPS. We'll read all ~60 meaningful lines
in four pieces.

### 2a. The shape of the tile, decided at compile time

```metal
// Compile-time constants set by the host via -D:
//   SW         tile-of-tiles edge length (threadgroup is SW x SW simdgroups)
//   SIMD_TILE  simdgroups own a SIMD_TILE x SIMD_TILE grid of 8x8 matrices
//   TILE_K     reduction tile = TILE_K * 8 elements

#include <metal_simdgroup_matrix>
#include <metal_compute>
using namespace metal;

constant constexpr ushort BM = SW * SIMD_TILE * 8;  // threadgroup output rows
constant constexpr ushort BN = SW * SIMD_TILE * 8;  // threadgroup output cols
constant constexpr ushort BK = TILE_K * 8;          // reduction tile
```
— [`sync_copy.metal:9-20`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L9-L20)

Nothing about the tile shape is a runtime variable. The host passes `-D SW=2
-D SIMD_TILE=4 -D TILE_K=2` to the compiler, and every loop bound, array size, and
address calculation below becomes a constant. With the defaults: each threadgroup owns
a 64×64 tile of C, each of its 2×2=4 simdgroups owns a 32×32 sub-tile, and the K
dimension is consumed 16 columns at a time. Compile-time shapes are why the compiler
can fully unroll everything — a theme that returns when we look at the benchmark table.

The kernel signature carries the single most consequential line in the file:

```metal
kernel void __attribute__((max_total_threads_per_threadgroup(SW * SW * 32)))
matmul(
```
— [`sync_copy.metal:84-85`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L84-L85)

That attribute promises the compiler the threadgroup will never exceed 128 threads.
m5-gemm's README calls it "the single biggest practical win": with the thread count
pinned, the register allocator knows exactly how many registers each thread can have
and stops spilling accumulators to memory. Remember from stage 0: registers are the
real budget. This one line is the difference between the accumulators living in the
register file or bouncing through memory.

### 2b. Getting the tile on-chip: the cooperative load

Before any math, the threadgroup's 128 threads jointly copy the current slab of A and
B into threadgroup memory. "Cooperative" means no thread loads *its own* data — the
whole threadgroup loads the whole tile, spread evenly:

```metal
template <ushort rows, ushort cols, ushort nthreads>
inline void load_tile(
    const device float *src,
    uint src_stride,
    threadgroup float *dst,
    ushort tid)
{
  constexpr ushort per_thread = (rows * cols + nthreads - 1) / nthreads;

  // Use vec4 loads when alignment permits.
  if ((cols % 4) == 0 && (per_thread % 4) == 0) {
    constexpr ushort cols4 = cols / 4;
    constexpr ushort total4 = rows * cols4;
    auto src4 = reinterpret_cast<const device float4 *>(src);
    auto dst4 = reinterpret_cast<threadgroup float4 *>(dst);
    uint stride4 = src_stride / 4;
#pragma clang loop unroll(full)
    for (ushort i = 0; i < total4; i += nthreads) {
      ushort idx = i + tid;
      if (idx >= total4) break;
      ushort r = idx / cols4;
      ushort c = idx - r * cols4;
      dst4[idx] = src4[uint(r) * stride4 + c];
    }
    return;
  }
```
— [`sync_copy.metal:46-71`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L46-L71)
(a scalar fallback follows for shapes that don't divide by 4)

Two things to notice:

- **Consecutive threads touch consecutive addresses** (`idx = i + tid`). That's
  coalescing: the memory system serves one wide transaction instead of 128 scattered
  ones. Same instinct as CUDA, and it matters just as much here.
- **`float4` when possible.** One vector load moves 16 bytes; with the default tile
  (64×16 floats = 1024 elements, 128 threads) each thread issues exactly two `float4`
  loads. The whole staging step is ~256 instructions across the threadgroup.

### 2c. The K-loop: stage, sync, multiply, sync

The heart of the kernel. March along the reduction dimension one BK-wide slab at a
time:

```metal
  uint k_tiles = k / BK;
  for (uint l = 0; l < k_tiles; l++) {
    uint k_off = l * BK;
    load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg, tid_in_tg);
    load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg, tid_in_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort2 simd_origin = ushort2(t_tg_pos.y, t_tg_pos.z) * (8 * SIMD_TILE);
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++)
        simdgroup_multiply_tile<TILE_K>(
            A_tg, B_tg,
            simd_origin + ushort2(i * 8, j * 8),
            acc[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
```
— [`sync_copy.metal:113-128`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L113-L128)

The first `threadgroup_barrier` says "nobody multiplies until the tile is fully
staged"; the second says "nobody overwrites the tile until everybody has finished
multiplying it." On Apple hardware these barriers cost ~2 cycles (stage 0's table) —
cheap enough that the sync itself is not the problem. The problem, which section 3
addresses, is that **loads and math take turns**: while the threadgroup is staging tile
`l+1`, its ALUs are idle.

The inner multiply is where the "tensor core" ops finally appear:

```metal
template <ushort DIM>
inline void simdgroup_multiply_tile(
    threadgroup float *A,
    threadgroup float *B,
    ushort2 c_pos,
    thread simdgroup_float8x8 &acc)
{
  simdgroup_float8x8 A_simd, B_simd;
#pragma clang loop unroll(full)
  for (ushort i = 0; i < DIM * 8; i += 8) {
    simdgroup_load(A_simd, A, DIM * 8, ulong2(i, c_pos.y));
    simdgroup_load(B_simd, B, BN,      ulong2(c_pos.x, i));
    simdgroup_multiply_accumulate(acc, A_simd, B_simd, acc);
  }
}
```
— [`sync_copy.metal:22-36`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L22-L36)

Each simdgroup holds a 4×4 grid of `simdgroup_float8x8` accumulators — its 32×32
patch of C — **in registers, for the entire kernel**. That's the reuse pyramid's top
level: every element of C is written to DRAM exactly once, at the very end. Count the
register cost, though: 16 accumulators × 64 floats = 1024 floats per simdgroup of
accumulator state alone. The README notes that doubling `SIMD_TILE` to 8 (64
accumulators) spilled and ran **10× slower**. Register pressure isn't a footnote on
this hardware; it's the wall.

### 2d. The epilogue

```metal
  if (c_col0 < m && c_row0 < n) {
    simdgroup_float8x8 c_simd;
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++) {
        ulong2 pos = ulong2(c_col0 + i * 8, c_row0 + j * 8);
        simdgroup_load(c_simd, C, m, pos);
        simdgroup_multiply(c_simd, c_simd, simdgroup_float8x8(beta));
        simdgroup_multiply_accumulate(c_simd, acc[i][j], simdgroup_float8x8(alpha), c_simd);
        simdgroup_store(c_simd, C, m, pos);
      }
  }
```
— [`sync_copy.metal:133-143`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L133-L143)

This computes the full BLAS contract `C = α·AB + β·C` — the scale-and-add happens
here, fused into the store, rather than in a second kernel that would re-read all of C
from DRAM. That's stage 0's lesson 1 (fusion beats extra passes) in its smallest form,
and stage 3 will show MLX generalizing exactly this spot into pluggable "epilogues"
(bias, activation) on the same GEMM body.

## 3. Double buffering: hiding the load behind the math

`sync_copy_db.metal` is the same kernel with one structural change: two sets of
threadgroup buffers, and the K-loop prefetches the *next* tile while computing on the
*current* one.

```metal
  threadgroup float A_tg[2][BM * BK];
  threadgroup float B_tg[2][BK * BN];
```
— [`sync_copy_db.metal:87-88`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy_db.metal#L87-L88)

```metal
  // Prologue: load tile 0 into buffer 0.
  load_tile<BM, BK, NTHREADS>(A + tg_row * k, k, A_tg[0], tid);
  load_tile<BK, BN, NTHREADS>(B + tg_col,     m, B_tg[0], tid);

  ushort cur = 0;
  ushort2 simd_origin = ushort2(t_tg_pos.y, t_tg_pos.z) * (8 * SIMD_TILE);

  for (uint l = 0; l < k_tiles; l++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ushort nxt = ushort(1) - cur;
    if (l + 1 < k_tiles) {
      uint k_off = (l + 1) * BK;
      load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg[nxt], tid);
      load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg[nxt], tid);
    }
    threadgroup float *A_use = A_tg[cur];
    threadgroup float *B_use = B_tg[cur];
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++)
        simdgroup_multiply_tile<TILE_K>(
            A_use, B_use,
            simd_origin + ushort2(i * 8, j * 8),
            acc[i][j]);
    cur = nxt;
  }
```
— [`sync_copy_db.metal:102-126`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy_db.metal#L102-L126)

Compare against the loop in 2c. What changed:

- The loads for tile `l+1` are issued **before** the math on tile `l`, into the other
  buffer. The loads and the multiplies have no data dependence on each other, so the
  hardware scheduler can interleave them — while a simdgroup waits on a load, it can
  run multiply-accumulates, and vice versa.
- **One barrier per iteration instead of two.** The single barrier at the top of the
  loop says "the prefetch that was in flight has landed, and everyone's done computing
  on the buffer we're about to overwrite" — both conditions at once, because the two
  buffers alternate roles.
- The file's own comment is honest about what this is: *"Without an async-DMA primitive
  this isn't true overlap, but it gives the scheduler enough independent instructions
  to hide threadgroup memory latency through ILP."* The same 128 threads still issue
  both the loads and the math; there's no separate copy engine. It's instruction-level
  parallelism, not a DMA engine.

**What it costs:** 2× the threadgroup memory (four buffers instead of two). Threadgroup
memory is 32 KB per group; doubling your footprint can halve how many threadgroups fit
on a core at once (occupancy), which reduces the machine's other latency-hiding
mechanism — having a different threadgroup's work to run. Double buffering trades
occupancy for ILP. Whether that trade wins depends on the problem size, which is
exactly what the numbers show.

## 4. What the numbers say

From [m5-gemm's README](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/README.md)
— measured on an M5 Max (40-core GPU), fp32, best of 3×5 runs. Their data, credited,
not re-measured here:

| Size | sync_copy | sync_copy_db | MPS |
|---:|---:|---:|---:|
| 1024² | 3.3 TF | **10.7 TF** | 2.9 TF |
| 2048² | 7.7 TF | **9.0 TF** | 5.2 TF |
| 4096² | **13.5 TF** | 13.1 TF | 11.7 TF |
| 8192² | 13.0 TF | 12.7 TF | **13.5 TF** |

Read the flips:

- **Small (1024²): double buffering wins 3×.** The matrix is small, so the fixed costs
  — kernel launch, first touch of DRAM — are a big fraction of total time, and the
  prologue-prefetch pipeline hides them. This regime is where LLM decode lives, by the
  way: lots of small, latency-dominated matmuls.
- **Medium (4096²): the simple kernel wins.** m5-gemm's explanation: the
  single-buffered loop is smaller, and the compiler unrolls a smaller loop better. A
  "worse" algorithm with better code generation beats a "better" algorithm — a pattern
  you will see constantly in this field.
- **Large (8192²): MPS wins, barely.** At this size everything is bandwidth bound
  (stage 0's one idea); every implementation converges on the same memory-traffic
  floor, and clever scheduling stops mattering.

One more number from that README worth internalizing: at 4096² this kernel *needs*
only ~6 GB/s of memory traffic against a measured ~500 GB/s ceiling — it's ~85× below
the bandwidth roof, i.e. solidly **compute-bound**. Tiling worked: it turned a
bandwidth-bound problem into a compute-bound one. That is the entire purpose of the
reuse pyramid in section 1.

## 5. The ghost: `simdgroup_async_copy`

The essay this all descends from used something better than cooperative loads: a real
asynchronous DMA intrinsic, Apple's closest analogue to CUDA's `cp.async`. Apple never
documented it, but the compiler exposed it, and Medley's original kernel declared it by
hand via inline assembly linkage:

```metal
thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d(
  ulong,               // sizeof(element)
  ulong,               // alignof(element)
  threadgroup void *,  // dst
  ...
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

void __metal_wait_simdgroup_events(
  int, // len(events)
  thread _simdgroup_event_t**
)
  __asm("air.wait_simdgroup_events");
```
— [`async_copy.metal:9-28`](https://github.com/0xekez/metal-matmul/blob/04e80810bbf7ba96ebe26ff84a346d179ee50888/async_copy.metal#L9-L28), abridged

and used it in the K-loop — note `if (s_pos==0)`: a *single simdgroup* kicks off the
copy for the whole threadgroup, fires both transfers, and waits on the events:

```metal
    if (s_pos==0) {
      thread _simdgroup_event_t* events[2];
      events[0] = simdgroup_async_copy<TILE_K*8,SW*SIMD_TILE*8>(
        A, a_pos, ushort2(k,n), A_tg);
      events[1] = simdgroup_async_copy<SW*SIMD_TILE*8,TILE_K*8>(
        B, b_pos, ushort2(m,k), B_tg);
      __metal_wait_simdgroup_events(2,events);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
```
— [`async_copy.metal:101-115`](https://github.com/0xekez/metal-matmul/blob/04e80810bbf7ba96ebe26ff84a346d179ee50888/async_copy.metal#L101-L115), abridged

The essay's counterintuitive finding was that this single-simdgroup arrangement was the
*fastest* way to load — the copy hardware does the moving, so parallelizing the ask
bought nothing.

**Why you can't use it:** the Metal 4 compiler (macOS 26+) rejects all `__asm("air.*")`
declarations outright. The technique is dead on current toolchains — m5-gemm exists
precisely because of that, replacing the DMA with the cooperative load + double
buffering you just read. It's kept in the ladder because the *concept* (a copy engine
running independently of the ALUs, completion signaled by events) is how CUDA's
`cp.async`/TMA works, how MLX's steel kernels once staged tiles, and quite possibly
how a future public Metal API will look. You'll recognize it instantly in stage 4's
metal-flash-attention, which documents an M1-era hardware bug in this same intrinsic.

## 6. Optional: run it yourself

Apple Silicon + Python. m5-gemm's harness compiles at runtime — no Xcode needed:

```sh
cd code/m5-gemm
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python matmul.py --dim 4096 --trials 5    # both kernels vs MPS
.venv/bin/python bandwidth.py                        # your machine's real DRAM ceiling
```

Your table will differ from the one above — the M5 Max has ~4× the cores and bandwidth
of a base M5, and the crossover points move with that ratio. Seeing *where* they move
on your machine is the best exercise this stage has.

## Done when

- You can draw the reuse pyramid (device → threadgroup → registers → 8×8 op) and say
  what gets copied at each level and why.
- You can explain the double-buffering trade from memory: what it costs (2× threadgroup
  memory → occupancy), what it buys (loads off the critical path via ILP), and why the
  winner flips with matrix size.
- You can say what `max_total_threads_per_threadgroup` does and why it mattered more
  than any algorithmic change.
- You can say what `simdgroup_async_copy` was, why a single simdgroup issuing the copy
  was fastest, and why the intrinsic no longer compiles.

Next: [Stage 3: How production factors it](stage-3-steel.md)
