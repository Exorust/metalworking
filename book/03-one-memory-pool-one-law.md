# One memory pool and the one law: bandwidth

## Where we are

Chapter 2 opened the core and found a large register file next to a small scratch
memory. Below the core sits the memory that every threadgroup reads its inputs from
and writes its results to. On a discrete NVIDIA card that memory is on the card,
and the host's memory is on the other side of a bus. On Apple Silicon there is one
pool, and it is shared with the CPU. This chapter measures what that buys and what
it costs, and the cost turns out to be the number that decides most kernel designs
on this platform.

## One pool

Apple Silicon has one physical memory, the same LPDDR chips at the same physical
addresses, used by the CPU and the GPU alike. There is no device memory and no host
memory, so there is no transfer between them. A `MTLBuffer` and a NumPy array can
be views of the same bytes; Chapter 6 shows how a buffer is made.

CUDA has something called unified memory, `cudaMallocManaged`, and on a discrete
card it is not this. There it is a software illusion over the bus, with pages
migrating back and forth. Here it is the hardware. Three things follow.

- **Capacity.** A Mac Studio configuration holds hundreds of gigabytes of memory
  that the GPU can address. Models that need to be sharded across several cards
  in CUDA run on one Apple chip, which is the main reason people who run
  language models locally care about this platform. The usable budget is smaller
  than the installed RAM: macOS caps how much of it the GPU may wire, and a
  `sysctl` raises the cap (Chapter 25). GPU allocations also compete with the
  operating system and everything else running.
- **No transfer engineering.** There is no `cudaMemcpy`, no pinned host memory,
  and no choreography of streams to overlap copies with compute. A whole genre of
  CUDA optimization does not exist here.
- **Cheap cooperation between CPU and GPU, with a cost that stays.** Data is free
  to share, but synchronizing is not. When the CPU waits for a result and has
  queued nothing behind it, the GPU sits idle while the CPU wakes, reads, and
  encodes the next batch of work. This is why the framework in Part 5 records
  operations and runs them only when something forces evaluation, and why
  halving the number of forced waits per step is a measured win in Part 7.

## What it costs

What Apple's chips give up for this design is bandwidth. A base M5 moves about
153 GB/s and an M5 Max about 614 GB/s. A datacenter GPU pairs its arithmetic with
HBM at several terabytes per second. The achievable fraction of the Apple figure
is lower than the spec: a STREAM-style probe on an M5 Max measured about 516 GB/s
for a copy, which counts the bytes read and the bytes written, about 84 percent of
the spec, and about 495 GB/s for a read-only sweep. Use the measured figure as the roof when you report a percentage of peak.

Combined with the modest clocks from Chapter 1, this gives the closest thing this
book has to a law: on Apple Silicon, most of the kernels you will write are
bandwidth bound, not compute bound. The win usually comes from touching less memory, through fusion, tiling for
reuse, and not doing work at all, rather than from arithmetic tricks. The
exceptions are large matrix multiply, and kernels that reach neither limit
because they wait on dispatch overhead or stalls. For language-model inference at
batch size one, memory bandwidth predicts tokens per second during generation
almost linearly across chip tiers. Read the bandwidth line of the spec sheet before
you profile anything.

## The number that decides: arithmetic intensity

To say which limit a kernel is nearer you need one number for the kernel and one
for the machine. The kernel's number is its arithmetic intensity: floating point
operations per byte of memory traffic. The machine's number is its ridge point:
the intensity at which the bandwidth ceiling and the compute ceiling cross. Below
the ridge the bandwidth ceiling is the tighter bound, above it the compute ceiling
is. The ridge says which ceiling applies; it does not say the kernel reaches
either one. A kernel that is far from both is overhead bound or stalled, and
Chapter 10 shows how to tell.

The ridge point is a division. Take the M5 Max and its 32-bit matrix-multiply
path. A well-written matrix multiply reaches 13.5 TFLOPS on it, which Part 4
shows, and that measured figure is the practical compute ceiling for this path.
The spec bandwidth is 614 GB/s.

    13.5 TFLOPS / 614 GB/s = about 22 FLOPs per byte

Use the measured 516 GB/s instead of the spec and the ridge moves to about 26
FLOPs per byte. Both values come from one measurement and one spec figure, and both apply to
this precision and path; other precisions have their own ceilings (Chapter 10).
The size matters more than the exact value: a kernel needs more than twenty
operations for every byte it moves before the compute ceiling binds. On a datacenter part, terabytes per second of HBM sit under
arithmetic that is not proportionally faster, so the crossing comes at a lower
intensity. That is what the law means in numbers: the ridge sits high, so more of
your kernels sit on the bandwidth side of it than your CUDA intuition expects.

Now put the canonical workloads against it. The intensities below are ideal counts,
every input byte read once and every output byte written once. They say what a
workload could reach, not what an implementation moves.

**Matrix multiply.** C = A × B at size n does 2n³ floating point operations on
3n² values. In 32-bit floats that is 12n² bytes if every value is touched once, so
the ideal intensity is 2n³ divided by 12n², which is n/6 FLOPs per byte. At
n = 4096 that is about 683, thirty times the ridge. Large matrix multiply is the
rare workload on this machine that is not inherently bandwidth bound. But no
kernel touches each value once. A kernel with no tiling issues n loads for every
input value. The tiled kernel this book follows computes a 64 by 64 output patch
per threadgroup, so each threadgroup loads a 64-row strip of A and a 64-column
strip of B: 2 × 64 × 64 × n operations against 2 × 64 × n × 4 bytes requested, which
is 16 FLOPs per byte requested, below the ridge. Tiling raises the intensity of
requested traffic from about 1 to a quarter of the tile edge, and the caches
between the cores and the memory absorb part of the repeated requests, because
several threadgroups ask for the same strip. The measured result is that the
kernel reaches 13.5 TFLOPS at 4096 by 4096, which is the compute ceiling, so at
that size it is compute bound in practice. At 1024 by 1024 the same kernel is not,
and Chapter 14 shows why.

**Everything elementwise.** An add reads two 4-byte values and writes one to do a
single operation: 1 FLOP per 12 bytes, an intensity of about 0.1, and no amount of
cleverness inside the kernel changes the count. The only lever is not doing the
traffic at all, which is why fusion, Chapter 15, is the most profitable technique
in this book.

**Decode.** Generating one token of a dense language model at batch size one
multiplies every weight matrix by one vector. A matrix-vector product does one
multiply and one add per weight, 2 operations, and a 16-bit weight is 2 bytes, so
the intensity is about 1 FLOP per byte, ignoring the vector itself, which is small
next to the matrix. That is twenty times below the ridge: generation speed tracks
the bandwidth spec, and storing weights in fewer bits, Chapter 23, is a speed
optimization and not only a memory-size one. The prompt-processing half of
inference is the opposite shape, a big matrix multiply, and Chapter 22 separates
the two workloads.

## The sixty-second arithmetic

Before optimizing any kernel: count the bytes it must move, count the operations
it must do, divide, and compare with the ridge for that precision and path. On the
bandwidth side, where most kernels sit, arithmetic cleverness is wasted; the
levers are fusion, 16-bit or quantized traffic, and deleting work. On the compute
side, the techniques of Part 4, register blocking and double buffering, earn their
keep. Then measure, because the calculation says which ceiling applies and only a
measurement says whether the kernel is near it. Chapter 10 turns this into a chart.

## Check yourself

1. An fp16 matrix-vector product multiplies a 4096 by 4096 weight matrix by one
   vector. What is its ideal arithmetic intensity, which side of the M5 Max ridge
   is it on, and what is the lower bound on its time at the measured read
   bandwidth?
2. The tiled matrix multiply reaches 13.5 TFLOPS at 4096 by 4096. Why does the
   ideal count n/6 not establish that on its own, and what does?
3. Why does the chapter say that the ridge point on a datacenter GPU is lower,
   and what does a lower ridge mean for a CUDA engineer's instincts?

### Answers

1. 2 operations per weight over 2 bytes per weight is 1 FLOP per byte, far below
   the ridge of about 22 to 26, so the bandwidth ceiling applies. The weights are
   4096 × 4096 × 2 bytes, which is 32 MiB; at about 495 GB/s that is a lower
   bound of about 68 microseconds, and the measured time can only be longer.
2. Because n/6 counts each input byte once, and the kernel requests each strip of
   A and B from many threadgroups: its requested intensity is about 16 FLOPs per
   byte, below the ridge. What establishes the result is the measurement, 13.5
   TFLOPS against a 13.5 TFLOPS ceiling, which means the caches absorbed enough
   of the repeated requests for the compute ceiling to bind.
3. Because HBM supplies terabytes per second under arithmetic that is not
   proportionally faster, so fewer operations per byte are needed before the
   compute ceiling is the tighter bound. A CUDA engineer is used to kernels
   crossing to the compute side sooner than they do here, which is why more
   kernels on Apple Silicon sit on the bandwidth side than that intuition
   predicts.

The law says which ceiling limits a kernel, and the next chapter is about the
other requirement, keeping the core's arithmetic units busy at all, which has its
own constants and its own reason to prefer 16-bit values.

Next: [Keeping the core busy: occupancy, stalls, and F16](04-keeping-the-core-busy.md)
