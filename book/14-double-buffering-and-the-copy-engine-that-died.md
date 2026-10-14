# Double buffering, and the copy engine that died

## Where we are

You finished reading the kernel in Chapter 13: 13.5 TFLOPS at 4096 by 4096, at
the practical compute ceiling. Coming from CUDA, you ask the next question of any
tiled loop: the load of the next slab and the multiply of the current one are
independent, so why not overlap them? The answer on this platform comes in
two parts. There is a double-buffered version of the kernel, and it wins at some
sizes and loses at others, which is the cleanest measured demonstration in this
book that there is no single fastest kernel. The hardware copy engine that a
CUDA kernel would use for the overlap exists on the chip, was once reachable, and
can no longer be reached; the shape of every current kernel follows from that.

## Double buffering: overlap through ILP, not DMA

You want the loop to load slab `l + 1` while it computes on slab `l`. Double
buffering allocates two staging buffers so that it can. In CUDA the same structure is usually fed by
`cp.async` or TMA, asynchronous copies that proceed while the threads compute.
Here the copy engine is not reachable, so the prefetch is ordinary loads and
stores issued by the same threads that do the math. What makes it work anyway is
that the prefetch and the current slab's multiply-accumulates have no data
dependence, so the scheduler can interleave them. That is the instruction-level
parallelism of Chapter 4, applied to the device-fetch latency of the next slab;
how much of it the compiler and the hardware actually overlap is not measured
here, only the result. It is software pipelining rather than hardware
pipelining.

The double-buffered loop differs from the single-buffered loop of Chapter 12 in
exactly three ways: two buffer sets, a prologue that loads slab 0, and one barrier per
iteration instead of two. The multiply, elided below, is the same fragment loop
as Chapter 12, run on `A_use` and `B_use`.

```metal
  threadgroup float A_tg[2][BM * BK];
  threadgroup float B_tg[2][BK * BN];
  ...
  // Prologue: load tile 0 into buffer 0.
  load_tile<BM, BK, NTHREADS>(A + tg_row * k, k, A_tg[0], tid);
  load_tile<BK, BN, NTHREADS>(B + tg_col,     m, B_tg[0], tid);

  ushort cur = 0;
  ...
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
    ...
    cur = nxt;
  }
```
Source: [`sync_copy_db.metal:87-126`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy_db.metal#L87-L126), abridged

The single barrier covers two hazards on two different buffers. At the top of
iteration `l`, "the prefetch of slab `l` into `cur` has landed" and "every
simdgroup has finished reading slab `l - 1` from `nxt`, which the next prefetch
will overwrite" are both true after one barrier, because the writes that fill
`cur` and the reads that drained `nxt` both happened before it. So the two
hazards of Chapter 2 are covered by one barrier per iteration; the first
iteration relies on the prologue, and the last issues no prefetch. The kernel's own header comment states the mechanism:
without the copy engine there is no true overlap, only the same threads issuing
independent loads and math for the scheduler to interleave.

![Three timelines comparing single-buffered serial load/compute, double-buffered overlap through ILP, and the dead DMA era's true copy-engine overlap](../diagrams/double-buffering.svg)

*What the restructuring buys, schematically. The ticks are threadgroup
barriers, one per iteration in the double-buffered loop (the single-buffered
lane shows only the post-load ones); the ghosted third lane is the hardware
overlap the platform lost with `simdgroup_async_copy`.*

The cost is the one Chapter 4 priced. Two buffer sets double the threadgroup's
scratch footprint, 16 KB instead of 8 KB for this kernel, which cuts the
scratch-limited bound on resident threadgroups from 7 to 3 (60 / 8 and 60 / 16,
rounded down), 12 simdgroups, below the saturation point of Chapter 4. Double buffering
trades one latency-hiding mechanism, occupancy, for another, independent work
inside each threadgroup. Whether the trade wins depends on which one the kernel
was short of.

## Three regimes, three winners

So which one was this kernel short of? The author measured the single-buffered
kernel, the double-buffered kernel and MPS at four sizes on an M5 Max, 32-bit, best of 3 runs each averaging 5 trials.

| Size | sync_copy | sync_copy_db | MPS |
|---:|---:|---:|---:|
| 1024² | 3.3 TF | 10.7 TF | 2.9 TF |
| 2048² | 7.7 TF | 9.0 TF | 5.2 TF |
| 4096² | 13.5 TF | 13.1 TF | 11.7 TF |
| 8192² | 13.0 TF | 12.7 TF | 13.5 TF |

![Three Metal GEMM kernels benchmarked across matrix sizes, showing the fastest one changes with the problem size](../gemm-ladder.gif)

Each size regime has a different winner. The timings tell you which kernel
wins; the author's explanations of why are hypotheses from differential
benchmarking, not isolated measurements.

At 1024 by 1024, the double-buffered kernel is 10.7 / 3.3, about 3.2 times the
single-buffered one, and 10.7 / 2.9, about 3.7 times MPS. The author attributes
this to fetch latency that the prefetch hides, and since the GPU timestamps
exclude launch cost, memory latency is the candidate. Small matrix multiplies
are the shape that batched and speculative decoding produce (Chapter 22).

At 4096 by 4096 the simpler kernel wins by a little, 13.5 against 13.1, because
its smaller loop unrolls better; compiler quality matters more than the
algorithm here.

At 8192 by 8192 all three converge, 13.0, 12.7 and 13.5, and MPS takes the lead
back; the author attributes the convergence to the memory system, and whatever
the mechanism, the kernels are no longer separated by their inner loop.

The general result is that the ranking is non-monotonic: no single kernel is
fastest at every size. Double buffering wins in some regimes and loses in others,
which is why production libraries dispatch different variants per shape
(Chapter 17) instead of shipping one kernel.

The README's section "things that did not help" is worth reading for its method
(Chapter 9): offline compilation with `-O3 -ffast-math`, measured identical to
runtime compilation, which the specification predicts, since runtime compilation
already defaults to fast math and the documented optimization levels are `-O2`
and `-Os`; bigger simdgroup tiles, the 10 times slower spill of Chapter
12; and `SW` = 3 threadgroups. The author measured these negative results and
published them, which is the norm that the failures in Chapter 25 argue the whole
ecosystem should follow.

## The copy engine that died

Reading the double-buffered loop, you would ask where `cp.async` is. The answer
is a short history. Apple's hardware has a copy engine that moves
tiles from device memory to threadgroup memory while the arithmetic units work,
with completion signaled by events: the same idea as `cp.async` and TMA. Apple
never exposed it in the Metal Shading Language. The compiler exposed it anyway,
and the community found it: an intrinsic could be declared by its internal AIR
symbol name through inline-assembly linkage.

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
Source: [metal-matmul `async_copy.metal:9-28`](https://github.com/0xekez/metal-matmul/blob/04e80810bbf7ba96ebe26ff84a346d179ee50888/async_copy.metal#L9-L28), abridged

Then you used it: start the copies, wait on the events, barrier, compute. The
pinned kernel does not overlap the copy with the previous slab's math; its gain
was that the copy engine did the moving instead of the threads. The
original of this book's matrix-multiply lineage, Zeke Medley's metal-matmul and
its essay, was built on it.

```metal
    if (s_pos==0) {
      thread _simdgroup_event_t* events[2];
      events[0] = simdgroup_async_copy<TILE_K*8,SW*SIMD_TILE*8>(
        A,
        a_pos,
        ushort2(k,n),
        A_tg);
      events[1] = simdgroup_async_copy<SW*SIMD_TILE*8,TILE_K*8>(
        B,
        b_pos,
        ushort2(m,k),
        B_tg);
      __metal_wait_simdgroup_events(2,events);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
```
Source: [`async_copy.metal:101-115`](https://github.com/0xekez/metal-matmul/blob/04e80810bbf7ba96ebe26ff84a346d179ee50888/async_copy.metal#L101-L115)

Read `if (s_pos==0)`: one simdgroup issues the copy for the whole threadgroup,
fires both transfers, waits on the events, and everyone else waits at the
barrier. The essay's counterintuitive measurement was that this was the fastest
arrangement, because the copy engine does the moving and spreading the request
across simdgroups bought nothing. CUDA's evolution toward TMA, where one thread
issues a bulk tensor copy, reached the same conclusion with official support.
metal-flash-attention built its tile staging on the same intrinsics, and
documents a suspected M1-era hardware bug: for certain configurations, an async
copy whose result is never read could freeze the GPU until reboot. That is the risk of building on an undocumented interface.

Metal 4, which shipped with macOS 26, removed that access; the m5-gemm README
names the frontend it tested, `metalfe-32023.883`. The compiler rejects every
`__asm("air.*")` declaration with the error "illegal string literal in 'asm'",
and workarounds at the intermediate-representation level crash the backend
compiler. There is no public replacement; Metal 4's cooperative tensors ship a
generic layout interface, not a copy primitive. The m5-gemm README documents the
failed attempts to resurrect it. Hence the two successor kernels you have read:
cooperative loads for the staging, Chapter 11, and double buffering to win back
some of the overlap through instruction-level parallelism, above.

You have two reasons to keep the dead code in mind. The concept outlives the
interface: an asynchronous copy with a completion you wait on is the idea behind
`cp.async` groups and TMA's transaction barriers, and how metal-flash-attention's
generated kernels staged tiles. It is plausibly how a future public Metal API
will look, and you will recognize it at a glance. It is also a
lesson in platform risk: every project that built on the undocumented intrinsic,
the essay, metal-flash-attention, early steel, carried a breakage risk that
Metal 4 made real. The cheap wins of Chapter 25 run the same trade knowingly, with
undocumented `sysctl` settings and private frameworks, and this kernel shows what the
downside looks like.

## Where the matmul stands

The example now has a family: at 1024 by 1024 the double-buffered kernel wins by
about 3 times, at 4096 by 4096 the single-buffered kernel holds 13.5 TFLOPS, and
at 8192 by 8192 the library catches up at 13.5. The DMA-staged original no longer
compiles. Inside the loop, you have used up the techniques this book covers; the
remaining gains are in what surrounds it, which is the next chapter.

## Review the content

1. The double-buffered kernel uses one barrier per iteration where the
   single-buffered kernel uses two. Which two hazards does the one barrier cover,
   and why is one enough?
2. At 1024 by 1024 the double-buffered kernel is about 3 times faster, and at
   4096 by 4096 it is slightly slower. What does the chapter say this pattern
   implies for a production library?
3. Why does a kernel written in 2024 against `simdgroup_async_copy` fail to
   compile on macOS 26, and what replaced the technique?

### Answers

1. "The prefetch of this iteration's slab into `cur` has landed" and "every
   simdgroup has finished reading the previous slab from `nxt`, which the next
   prefetch overwrites". They concern different buffers, but the writes and the
   reads both happened before the top of the iteration, so one barrier there
   covers both.
2. That no single kernel is fastest at every size, so a library should dispatch
   different variants per shape rather than ship one kernel; Chapter 17 shows MLX
   doing exactly that.
3. Metal 4's compiler rejects every `__asm("air.*")` declaration, the route
   the kernel used to reach the intrinsic, and the IR-level workarounds crashed
   the backend. Cooperative loads replaced the staging, and double buffering,
   through instruction-level parallelism, took back some of the overlap; how
   much is not measured against the dead kernel.

The loop is done, and what surrounds it is where the next gains are: the
operations before and after a matrix multiply that, done separately, each cost a
full pass over memory.

Next: [Fusion and epilogues](15-fusion-and-epilogues.md)
