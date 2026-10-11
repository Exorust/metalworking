# One memory pool and the one law: bandwidth

## Where we are

In Chapter 2 you opened the core and found a large register file next to a small
scratch memory. Every threadgroup you launch reads its inputs from the memory below the core
and writes its results there. On a discrete NVIDIA card that memory is on the card,
and the host's memory is on the other side of a bus. On Apple Silicon there is one
pool, and it is shared with the CPU. You can share a buffer with the CPU without copying it. What it costs
turns out to be the number that decides most kernel designs on this platform.

## One pool

What does one pool mean? Apple Silicon has one physical memory, the same LPDDR chips at the same physical
addresses, used by the CPU and the GPU alike. There is no device memory and no host
memory, so there is no transfer between them. A `MTLBuffer` and a NumPy array can
be views of the same bytes; Chapter 6 shows how a buffer is made.

CUDA has something called unified memory, `cudaMallocManaged`, and on a discrete
card it is not this. There it is a software illusion over the bus, with pages
migrating back and forth. Here it is the hardware, and that has three consequences.

The first is capacity. A Mac Studio configuration holds hundreds of gigabytes that
the GPU can address, so models that need several cards in CUDA run on one Apple
chip, which is the main reason people who run language models locally care about
this platform. The usable budget is smaller than the installed RAM: macOS caps how
much the GPU may wire, a `sysctl` raises the cap (Chapter 25), and GPU allocations
compete with everything else running.

The second is that you stop engineering transfers. There is no `cudaMemcpy`, no
pinned host memory, and no arrangement of streams to overlap copies with compute.
That class of CUDA optimization does not apply.

The third is cheap cooperation between CPU and GPU, with a cost that stays. Data is
free to share, but synchronizing is not. When the CPU waits for a result and has
queued nothing behind it, the GPU sits idle while the CPU wakes, reads, and encodes
the next batch of work. This is why the framework in Part 5 records operations and
runs them only when something forces evaluation, and why halving the number of
forced waits per step is a measured win in Part 7.

## What it costs

What do Apple's chips give up for this design? Bandwidth. A base M5 moves about
153 GB/s and an M5 Max about 614 GB/s. A datacenter GPU pairs its arithmetic with
HBM at several terabytes per second. The achievable rate is lower than the
spec: a STREAM-style probe on an M5 Max measured about 516 GB/s
for a copy, which counts bytes read and written, about 84 percent of
the spec, and about 495 GB/s for a read-only sweep. Use the measured figure as the roof when you report a percentage of peak.

Combined with the modest clocks from Chapter 1, this gives the closest thing this
book has to a law: on Apple Silicon, most of the kernels you will write are
bandwidth bound. The win usually comes from touching less memory, through fusion,
tiling for reuse and deleting work, rather than from arithmetic tricks. The
exceptions are large matrix multiply and kernels that reach neither limit because
they wait on dispatch overhead or stalls. At batch size one, memory bandwidth
predicts tokens per second during generation almost linearly across chip tiers,
so read the bandwidth line of the spec sheet before you profile anything.

## The number that decides: arithmetic intensity

Which ceiling are you under? Two numbers answer it. A kernel's number is its arithmetic intensity, floating point operations per
byte of memory traffic; the machine's is its ridge point, the intensity at which
the bandwidth and compute ceilings cross. Below the ridge the bandwidth ceiling
is the tighter bound, above it the compute ceiling. The ridge says which ceiling applies, not whether the kernel reaches it; a
kernel far from both is overhead bound or stalled, and Chapter 10 shows how to
tell.

The ridge point is a division. On the M5 Max's 32-bit matrix-multiply path, a
well-written kernel reaches 13.5 TFLOPS (Part 4), the practical compute ceiling
for this path, and the spec bandwidth is 614 GB/s.

    13.5 TFLOPS / 614 GB/s = about 22 FLOPs per byte

Use the measured 516 GB/s instead of the spec and the ridge moves to about 26
FLOPs per byte. Both use the
same measured compute figure, and both apply to this precision and path; other
precisions have their own ceilings (Chapter 10). A kernel needs more than about twenty-two to twenty-six
operations for every byte it moves before the compute ceiling binds.

Compare that with an H100: on its 32-bit path, about 67 TFLOPS over 3.35 TB/s,
the ridge is about 20, the same as here; on its 16-bit tensor-core path, about
989 TFLOPS over the same bandwidth, it is near 300. So the law is not that this
ridge is unusually high. It is that both ceilings are low in absolute terms and
bandwidth lowest of all, 516 GB/s against several terabytes per second. A
bandwidth-bound kernel takes about six times longer here for the same bytes, so
such kernels are a larger share of a model's time than your CUDA intuition
expects. Moving fewer bytes is the lever that pays most often.

Compare the ridge with matrix multiply, elementwise work and decode. The intensities below are ideal
counts, every byte touched once: what a workload could reach, not what an
implementation moves.

**Matrix multiply.** C = A × B at size n does 2n³ floating point operations on
3n² values. In 32-bit floats that is 12n² bytes touched once, so the ideal
intensity is n/6 FLOPs per byte (n/8 for the kernel this book follows, which
also reads C): about 683 or 512 at n = 4096, more than twenty times either
ridge. Large matrix multiply is the rare workload here that is not inherently
bandwidth bound.

But no kernel touches each value once. A kernel with no tiling issues n loads for every
input value: two operations per pair of 4-byte loads, 0.25 FLOPs per byte. The
tiled kernel this book follows computes a 64 by 64 output patch per threadgroup,
so each threadgroup loads a 64-row strip of A and a 64-column strip of B: 2 × 64 ×
64 × n operations against 2 × 64 × n × 4 bytes requested, which is 16 FLOPs per
byte requested, still below the ridge. Tiling raises the requested intensity from
0.25 to a quarter of the tile edge.

Requested is not compulsory. Each value of A
and B needs to come from memory once and C once each way, so the compulsory
traffic at 4096 by 4096 is 192 MiB (256 MiB for the kernel here, which also
reads C). In about 10 ms that is 20 to 26 GB/s, roughly 25 times below the
measured roof. The requested traffic, 4,096 threadgroups times 2 MiB of strips, about 8
GiB in the same 10 ms, would be about 840 GB/s, above any roof, so the caches
must serve most of it. The measured 13.5
TFLOPS at 4096 by 4096 is the best rate on this path and the figure this book
uses as its practical ceiling; the compulsory count is what shows the kernel is
not bandwidth bound, and Chapter 10 does the full accounting.

**Everything elementwise.** An add reads two 4-byte values and writes one to do a
single operation: 1 FLOP per 12 bytes, an intensity of about 0.1. Nothing inside
the kernel changes that count; the only lever is not doing the traffic, which is
why fusion (Chapter 15) is the most profitable technique in this book.

**Decode.** Generating one token of a dense language model at batch size one
multiplies every weight matrix by one vector. A matrix-vector product does 2 operations
per weight, and a 16-bit weight is 2 bytes, so the intensity is about 1 FLOP per
byte, ignoring the small vector. That is more than twenty times below the ridge: generation speed tracks
the bandwidth spec, and storing weights in fewer bits, Chapter 23, is a speed
optimization and not only a memory-size one. Prompt processing is the opposite shape, a big
matrix multiply, and Chapter 22 separates the two.

## The sixty-second arithmetic

Before you optimize a kernel, count the bytes it must move and the operations it
must do, divide the second by the first, and compare the result with the ridge for
that precision and path. On the
bandwidth side, where most kernels sit, arithmetic cleverness is wasted; the
levers are fusion, 16-bit or quantized traffic, and deleting work. On the compute
side, the techniques of Part 4, register blocking and double buffering, pay off.
Then measure, because only a measurement says whether the kernel is near its
ceiling. Chapter 10 turns this into a chart.

## Check yourself

1. An fp16 matrix-vector product multiplies a 4096 by 4096 weight matrix by one
   vector. What is its ideal arithmetic intensity, which side of the M5 Max ridge
   is it on, and what is the lower bound on its time at the measured read
   bandwidth?
2. The tiled matrix multiply reaches 13.5 TFLOPS at 4096 by 4096. Why does the
   ideal count n/6 not establish that on its own, and what does?
3. How does the M5 Max ridge compare with an H100's, and why is moving
   fewer bytes still the lever that pays most often here?

### Answers

1. 2 operations per weight over 2 bytes per weight is 1 FLOP per byte, far below
   the ridge of about 22 to 26 (the 16-bit ceiling is higher, which only moves
   the ridge up), so the bandwidth ceiling applies. The weights are
   4096 × 4096 × 2 bytes, which is 32 MiB; at about 495 GB/s that is a lower
   bound of about 68 microseconds, and the measured time can only be longer.
2. Because n/6 counts each input byte once, and the kernel requests each strip of
   A and B from many threadgroups: its requested intensity is about 16 FLOPs per
   byte, below the ridge. What establishes the result is the compulsory count, about 20 GB/s against a
   roof of about 500, while the requested 840 GB/s exceeds any roof, so the
   caches must absorb most of the repeats; 13.5 TFLOPS is the best measured rate
   on the path, not a proof of saturated arithmetic units.
3. On the 32-bit path the ridges are about the same, 20 to 26 FLOPs per byte;
   on the H100's 16-bit tensor path the ridge is near 300. The law comes from
   absolute bandwidth: at 516 GB/s against 3.35 TB/s, a bandwidth-bound kernel
   takes about six times longer here for the same bytes, so bytes moved is the
   lever.

The law says which ceiling limits a kernel, and the next chapter is about the
other requirement, keeping the core's arithmetic units busy at all, which has its
own constants and its own reason to prefer 16-bit values.

Next: [Keeping the core busy: occupancy, stalls, and F16](04-keeping-the-core-busy.md)
