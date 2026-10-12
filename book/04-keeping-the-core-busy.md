# Keeping the core busy: occupancy, stalls, and F16

## Where we are

You have the law from Chapter 3: most kernels on this chip wait for memory, not for
arithmetic. That tells you which ceiling limits a kernel. It does not decide whether
the arithmetic units are busy while the kernel runs, which is a separate
requirement with its own numbers. On NVIDIA you chase residency, thousands of
threads per SM, so that the scheduler always has something to issue while other
threads wait. The same mechanism exists here. Two constants differ: how many
resident simdgroups it takes to saturate a core, and how long a dependent pair of
instructions stalls at each precision. The second constant is why 16-bit values are
the local default, for a reason that has nothing to do with peak throughput.

## Occupancy: same word, moved constants

You already tune for occupancy, but this book counts it differently. Here it means
the number of simdgroups resident on a core at once: the pool of independent work
the scheduler hides latency with. CUDA usually expresses it as a fraction of the maximum residency; this book uses the absolute
count, because the measurements are reported that way. The limiters are the same
as in CUDA. Register use per thread and threadgroup memory per threadgroup divide
the core's resources among the threadgroups that want to be resident, and when the
resources run out, the next threadgroup waits.

The number to recalibrate is the saturation point. The arithmetic units reach full utilization at about
24 resident simdgroups per core, which is 24 × 32 = 768 threads, in the
metal-benchmarks measurement on M1- and M2-class chips. That is far below the
thousands of threads of residency an SM supports and that CUDA tuning advice
tells you to chase. In threadgroups of 128 threads, the working size from
Chapter 1, 24 / (128 / 32) = 6 resident threadgroups reach that point. Moderate
occupancy is enough to fill this machine.

The same measurement bears on register pressure. Twenty-four simdgroups is also the lowest occupancy that over-allocating
registers can produce on those chips: a kernel can use so many registers per
thread that only 24 simdgroups fit, but not fewer. So register pressure alone does
not push a core below the saturation point. Scratch can: a threadgroup at the
32 KB cap is the only one that fits in a core's 60 KB, which is 4 resident
simdgroups, and the double-buffered estimate for the kernel this book follows lands at
12. So can a lack of work: a grid too small to fill the cores, or a kernel whose
threadgroups are nearly done. Those are the cases where the stall numbers below
matter.

Apply the resource arithmetic to the matrix multiply this book follows, with
Chapter 2's figures. Each threadgroup stages 8 KB of inputs, so the core's roughly
60 KB of physical scratch memory holds at most 60 / 8 = 7 such threadgroups, which
is 7 × 4 = 28 simdgroups, past the saturation point. That is a scratch-only
estimate with approximate capacities, and it says scratch is not the binding limit
for this kernel. The register side cannot be settled from payload counts: the
accumulators are 16 KB per threadgroup, but the compiler's total allocation per
thread, which includes input fragments, addresses and loop state, is what the
residency calculation needs, and Chapter 26 shows how to read it from the compiled
code. The practical order is the one Chapter 2 gave: when a register-heavy kernel
is slow, suspect spills before occupancy, because the allocator cannot reduce
occupancy below the saturation point but spills cost a memory operation per use.

The `max_total_threads_per_threadgroup` attribute from Chapter 1 trades one limit
against the other.
Declaring a smaller maximum lets the compiler give each thread more
registers, which reduces spills at the price of fewer resident threadgroups, down
to the floor of 24 simdgroups. On this kernel the trade is worth it, because 6
threadgroups already saturate the core; Chapter 12 reads the attribute in place.

There is no `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on this platform, and
no scriptable profiler that reports achieved occupancy; Xcode's GPU capture shows
an occupancy estimate for one dispatch at a time. You reason from resource
arithmetic, threadgroup memory bytes and a register estimate, and then confirm by
measuring. Part 3 covers the measuring.

## The stall side: why precision matters here

When occupancy is low, what does a stall cost? Occupancy buys latency hiding, and the latency these measurements isolate is the
simplest kind: the gap between a dependent pair of arithmetic instructions. The
numbers come from the ALU Bottlenecks section of metal-benchmarks, measured on
M1- and M2-class chips with a shader setup its author calls sub-optimal. A later
run with a better setup measured 6 cycles for a dependent 32-bit fused
multiply-add and 3 for 16-bit. Trust the table for the shape, worse at low
occupancy and better in 16-bit, and the later run for the size of the gap, about
2 times. Use the fused multiply-add rows for this comparison; the 1.84-cycle
figure quoted elsewhere in the sources is the plain multiply row of the same table.

| Condition | 32-bit dependent chain | 16-bit dependent chain |
|---|---|---|
| 88 simdgroups per core, back-to-back dependent fused multiply-add | about 1.99 cycles per instruction | about 1.56 cycles |
| 4 simdgroups per core, back-to-back dependent fused multiply-add | about 11.3 cycles per instruction | about 3.9 cycles |

At 4 resident simdgroups, 11.3 / 3.9 is about 2.9 in this table, about 2 in the
corrected run: a 32-bit dependent chain runs two to three times slower per
instruction than a 16-bit one. At high occupancy the
ratio is 1.99 / 1.56, about 1.3. The 16-bit advantage is largest when the
core has the least parallelism to hide stalls with. The same table shows the
other way out: at 4 simdgroups, giving each thread four independent chains instead
of one brings the 32-bit figure from 11.3 to about 2.75 cycles. Independent work
inside a simdgroup hides most of the latency that other simdgroups would.

So the reason to prefer 16-bit values on this chip differs from the headline reason
on NVIDIA. There, fp16 and bf16 buy Tensor Core throughput, a peak-FLOPs
argument, on top of the storage saving every platform shares. Here the arithmetic
throughput tables show F16 and F32 tied, and the argument has three parts. The
first is shorter stalls when the computation is in 16-bit registers, as the table
above shows. The second is half the register payload for every value held in
16 bits, which can mean more resident simdgroups when registers are the binding
limit, and more headroom before the spilling cliff of Chapter 2. The third is half
the memory traffic, which under the law of Chapter 3 is often the whole result.

That is why "use F16 everywhere you can tolerate it" is the usual advice on this chip even where
the throughput tables say the two precisions are equal.

## The numerics rule that comes with it

Half-precision storage does not mean half-precision arithmetic. The attention and
inference kernels of Parts 5 and 6 store and move values in 16-bit and accumulate
in 32-bit: MLX's steel templates default their accumulator
type to `float` regardless of the data type, and the attention kernels keep their
running sums in 32-bit while streaming 16-bit keys and values. This limits the
growth of rounding error over long sums; it does not repair precision lost before
the sum.

This has a cost: the accumulators are 32-bit registers, so the stall
and payload advantages apply to the 16-bit operands, not to the accumulation
chain.

The 32-bit matrix multiply of Part 4 is 32-bit throughout by design, because
it is a 32-bit benchmark against the library.

Which half-precision formats does the rule cover? bf16 is supported, and the storage and
traffic advantages apply to it. Whether its arithmetic takes a 16-bit path depends
on the generation: on M1- and M2-class chips, metal-flash-attention widens bf16
operands to 32-bit in registers and its author describes the emulation cost.

fp8 had no native execution path on
M4-class hardware. Reverse engineering of Metal 4.1's tensor API, tested on an
M4 Max, found fp8 emulated in software at about 0.94 times fp16 throughput: it
halved the bytes without any speedup. On M5 the neural accelerators change this,
and block-scaled fp8 and fp4 paths ship with macOS 27's Metal; Chapter 28 covers
them.

16-bit integers get the same stall advantage over
32-bit integers. Index arithmetic in `ushort` is a standing micro-optimization in
production kernels, valid when the index provably fits in 16 bits, and you will
see `ushort` loop counters throughout the kernel in Part 4.

## Occupancy is a currency

If moderate occupancy already fills the core, what is the rest of it for? Treat
occupancy as something you spend rather than something you maximize. The second channel for hiding latency is
independent instructions inside one simdgroup's own stream, which the hardware
reorders freely. A kernel that issues two loads that do not depend on
each other, or a load and an unrelated multiply, hides latency without any help
from other simdgroups.

Double buffering, the recurring trade in Part 4, uses this second channel. It allocates two
staging buffers instead of one, which doubles the threadgroup's scratch footprint
and cuts the scratch-limited bound on residency from 7 threadgroups to 3 (60 / 8
and 60 / 16, rounded down), which is 12 simdgroups, below the 24 that saturate
the arithmetic units; that occupancy is what the trade spends, so that it can issue the next slab's loads while
the current slab's multiplies are still in flight. Which side of the trade wins depends
on problem size, and Chapter 14 has the measured table. For any change, ask what
occupancy it costs and what independent work it buys.

## Review the content

1. A kernel uses threadgroups of 256 threads. In the metal-benchmarks model, how
   many must be resident to saturate a core's arithmetic units, and how many
   simdgroups is that?
2. A kernel runs at 4 resident simdgroups with a long chain of dependent fused
   multiply-adds. Roughly how much slower per instruction is it in 32-bit than in
   16-bit, and what are the two ways to narrow the gap?
3. The attention kernels store and move 16-bit values but accumulate in 32-bit.
   Which of the three 16-bit advantages apply to the accumulation chain, and what
   does the 32-bit accumulator cost in register payload for a 32 by 32 patch?

### Answers

1. 256 / 32 = 8 simdgroups per threadgroup, and 24 / 8 = 3 resident threadgroups,
   which is 3 × 8 = 24 simdgroups and 3 × 256 = 768 threads. That is where
   arithmetic-unit utilization flattened on a synthetic kernel, not a guarantee
   for every kernel.
2. 11.3 / 3.9, about three times slower in the table and about twice in the
   corrected run. Raise occupancy, which brings the
   measured pair to 1.99 against 1.56, or give each thread independent work, which
   at four chains brings the 32-bit figure to about 2.75 cycles.
3. None of them: the accumulators are 32-bit registers, so the stall and payload
   advantages apply only to the 16-bit operands, and the traffic advantage applies
   to the operands that move. A 32 by 32 patch of 32-bit accumulators is
   32 × 32 × 4 = 4,096 bytes per simdgroup, against 2,048 bytes if it were 16-bit.

Part 1 ends with the two instruction-level facts that fit under neither bandwidth
nor occupancy and that shape the attention kernels: one exponent path and one
missing atomic.

Next: [The two asymmetries: fast exp2, emulated float atomics](05-the-two-asymmetries.md)
