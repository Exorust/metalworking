# Two hierarchies: what you launch and what runs it

## Where we are

This book takes one piece of work, a matrix multiply, from the library that ships
with the platform to a hand-written kernel that beats it, then to the production
framework's version of the same design, and then to attention. You know GPU programming
from the CUDA side. The job of Part 1 is to replace the hardware half of that
knowledge with the Apple version, one difference at a time. Before any number, you
need the vocabulary, because Apple uses one word, "core", for a different level
than NVIDIA does, and that one word causes a lot of early confusion.

## Two hierarchies, not one

There are two separate hierarchies, and they are easy to mix up because the words
look alike. One is software: the threads that a single kernel launch creates. You
choose its shape every time you launch. The other is hardware: the parts of the chip
that run those threads. You change that only by buying a different chip.

Here is the software hierarchy, top to bottom, with the CUDA name beside each level.

| Level | Metal name | CUDA name | What the threads in it share |
|---|---|---|---|
| All the threads of one launch | grid | grid | nothing; threadgroups are independent |
| A group of threads that cooperate | threadgroup | thread block | threadgroup memory and barriers |
| 32 threads that execute together | simdgroup | warp | registers, through shuffles |
| One execution of the kernel function | thread | thread | its own registers |

And the hardware hierarchy, top to bottom.

| Level | Apple name | NVIDIA name |
|---|---|---|
| The whole device | GPU | GPU |
| One independent unit of compute | GPU core | Streaming Multiprocessor (SM) |
| One arithmetic unit inside it | arithmetic unit | CUDA core |

![Two hierarchies side by side. Software, top to bottom: a thread, a threadgroup of four simdgroups with 32 KB of threadgroup memory, and a grid of threadgroups over device memory. Hardware, top to bottom: an arithmetic unit, a GPU core with a 208 KB register file, 60 KB of threadgroup memory and an 8 KB L1 cache, and a GPU of 10 to 40 cores over unified memory shared with the CPU. Each software level is scheduled onto the hardware level beside it.](../diagrams/architecture-levels.svg)

*Left: what one launch creates. Right: what runs it. Each row reads across. The
sizes drawn inside the core are the subject of the next chapter.*

Reading the two tables side by side shows the trap. In NVIDIA's
vocabulary, a "CUDA core" is the smallest thing on the right: one arithmetic unit,
of which an SM has many. In Apple's vocabulary, a "GPU core" is the middle thing:
the whole SM-sized unit, with its own register file, its own scratch memory, and
its own set of arithmetic units. When an Apple spec sheet says a chip has 40 GPU
cores, it means 40 units at the level of an SM, each running its own threadgroups. "Size"
here means the level in the hierarchy, not die area or throughput. A base M5 has
10 of them, and an M5 Max has 40.

There is no Apple datacenter part: the largest GPU Apple makes is in a
laptop or a desktop, and the design follows from that. The cores are wide and run
at low clocks, tuned for performance per watt rather than peak throughput.

## How a launch maps onto the chip

The two hierarchies meet at four points.

**A threadgroup runs on one core.** This is the reason its threads can share memory: the scratch memory a
threadgroup uses is physically inside the core it runs on. It is also the reason
threadgroups share nothing with each other. Two threadgroups in the same grid can
both read and write device memory, and integer atomic operations on device memory
can combine counts inside one dispatch (float atomics are emulated, Chapter 5),
but no threadgroup can read a finished total within that dispatch: there is no
grid-wide barrier and no promise that both are resident at the same time, so a
threadgroup must never wait for another one. Work that needs everything finished goes in the next dispatch.

**A core holds several threadgroups at the same time.** They divide the core's
registers and scratch memory between them. How many fit is a resource calculation,
not a launch parameter, and Chapter 4 does that calculation.

**The core schedules simdgroups, not threads.** A simdgroup is 32 threads that
execute the same instruction at the same time, and it is the unit the scheduler
picks from. A threadgroup of 128 threads is four simdgroups, numbered by
`[[simdgroup_index_in_threadgroup]]`. Kernels on this platform are usually designed
at the simdgroup level: in the matrix multiply this book follows, each simdgroup
owns its own patch of the output, and the threadgroup exists mostly so that its
four simdgroups can share one staging buffer.

**The grid is usually larger than the chip.** The threadgroups that do not fit at
launch wait, and each starts when a core has room. As in CUDA, a kernel cannot rely
on the order. The launch itself is one of two calls, one that counts the grid in
threadgroups and one that counts it in threads and can produce partial threadgroups
at the edge; Chapter 7 covers both.

## What the simdgroup gives you

Your warp intuition transfers, with one exception noted below. The width is 32,
the execution model is lockstep, and the simdgroup is where divergence costs you
and where threads exchange register values without going through memory.

| CUDA | Metal |
|---|---|
| warp | simdgroup |
| lane | `[[thread_index_in_simdgroup]]` |
| `__shfl_sync` and friends | `simd_shuffle`, `simd_shuffle_xor`, and so on |
| warp-level reduction | `simd_sum`, `simd_max`, and so on |
| `__syncwarp()` | `simdgroup_barrier`, with a memory flag that says what to make visible |
| warp-level MMA (`mma.sync`) | `simdgroup_matrix`, the same API shape; before M5 it runs on the regular pipes, not a matrix unit (Chapter 12) |

The exception: there is no Independent Thread Scheduling and there are no
`__syncwarp` masks to think about. The model is the classic one: if the 32 threads
take different sides of a branch, both sides execute, with the lanes on the other
side masked off. Where a branch depends on a function constant, a value bound when
the pipeline is built, Metal kernels remove it from the compiled code instead of
paying for it at run time; Chapter 8 shows how. A branch that depends on the data
still diverges in the usual way when the lanes disagree.

Shuffles are fast enough to use freely. The community's measurements put shuffle
bandwidth at 256 bytes per cycle per core, and the attention kernels in Part 6
reduce a row maximum across a simdgroup with shuffles plus a `max`, not through
memory.

## What the threadgroup gives you

The threadgroup is the level you size. Its threads share a scratch memory inside
the core and can wait for each other at a barrier, which is what a cooperative load
needs: many threads copy one tile into the shared buffer, all wait until the copy is
complete, each simdgroup reads its part, and all wait again before the next tile
overwrites the buffer. Size threadgroups in multiples of 32 so that every simdgroup
is full; 128 threads, four simdgroups, is the size of the matrix multiply this book
follows and of MLX's steel attention kernel.

The launch call that counts in
threads can make a partial threadgroup at the edge of the grid, the common case
where a simdgroup is not full; a threadgroup size that is not a multiple of 32 is
the other. Simdgroup operations, shuffles, reductions and `simdgroup_matrix`,
assume every lane takes part, so a short simdgroup breaks them rather than merely
wasting lanes.

The compiler allocates registers when it builds the pipeline, before it knows the
launch dimensions. Unless the kernel declares a maximum threadgroup size with the
attribute `max_total_threads_per_threadgroup`, the compiler must plan for the
largest threadgroup the hardware allows, and it has fewer registers per thread to
give.

CUDA's `__launch_bounds__` looks similar but works the other way around:
nvcc spends registers freely by default, and the bound caps them to raise
occupancy. Here the default is the stingy one, and the attribute raises each
thread's grant, which matters because kernels on this platform keep their
working set in registers. The hand-written matrix multiply and MLX's steel
kernels both carry it, its author calls it the kernel's single biggest practical
win, and Chapter 12 reads it in place.

## A rule for telling the two apart

If you can change it in your code, it is software: the grid, the threadgroup size,
the number of simdgroups. If you can change it only with a different chip, it is
hardware: the core, and the memories inside it. When a sentence in a Metal
document says "core", ask which hierarchy it belongs to. On Apple, it is always the
hardware one, and it is always the unit at the level of an SM.

## Review the content

1. A kernel is launched with a grid of 4096 threadgroups of 128 threads each, on a
   chip with 40 cores. How many simdgroups does one threadgroup contain, and what
   decides how many threadgroups run at the same time?
2. Suppose an NVIDIA spec sheet says an SM has 128 CUDA cores and an Apple spec
   sheet says a chip has 40 GPU cores. Which of the two "core" counts is the same
   kind of thing as a count of SMs?
3. Two threadgroups in the same grid need to combine their results. What can they
   use, and what must they never do?

### Answers

1. Four simdgroups, because 128 divided by 32 is 4. The number of threadgroups that
   run at once is a resource calculation per core, chiefly registers and scratch
   memory divided among resident threadgroups, times the 40 cores. That is an
   upper bound on what is resident; the rest wait.
2. The Apple count. An Apple GPU core is the unit at the level of an SM; an NVIDIA
   CUDA core is one arithmetic unit inside an SM.
3. They can write partial results to device memory for the next dispatch to
   combine, or update integer counters with atomics inside the dispatch, as long
   as nothing reads the total until the dispatch is over. They must never wait
   for each other, because there is no grid-wide barrier and no promise
   that both are resident at the same time.

The next chapter opens the core and measures what is inside: a register file nearly
as large as an SM's, next to a scratch memory far smaller, and what that ratio
does to kernel design.

Next: [The core's memories: 208 KB of registers, 32 KB of scratch](02-the-cores-memories.md)
