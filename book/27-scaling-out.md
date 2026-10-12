# Scaling out across machines

## Where we are

Every kernel you have timed so far ran inside one machine. You cannot add a
second card or an NVLink connection, because there is one GPU per Apple Silicon
machine (Chapter 6), and inside one machine the way to scale is to buy more
unified memory (Chapter 3). When your
model does not fit in one machine, you go the other direction: `mx.distributed`
runs it across several. With macOS 26.3 and Thunderbolt 5 the connection between
them is remote direct memory access at microsecond latencies, so the lack of
NVLink no longer ends the scaling story. The performance intuition you need was
already built in Chapter 22.

## What it is, against NCCL

If you have used NCCL, your first question is how the link compares. The CUDA
counterpart is NCCL plus the fabric underneath it, and in numbers: NVLink moves hundreds of gigabytes per second between GPUs in one
chassis; Thunderbolt 5 RDMA moves tens of gigabytes per second between chassis.
The Apple version is still worth having because of the memory arithmetic: each
node brings up to 512 GB of unified memory, so two Mac Studios hold a
trillion-parameter model resident in GPU-addressable memory, which no single
consumer NVIDIA card can do; a machine with such a card can still run the model
by paging weights through host memory, slowly. Kimi K2, a trillion-parameter model running across two M3 Ultra
machines through `mx.distributed`, is the public proof.

The pieces come from the MLX distributed documentation and session 233 of WWDC
2026, and the first one you choose is the backend. `ring` runs over TCP on any
network, works everywhere, and is slow. JACCL does RDMA over Thunderbolt 5, needs
macOS 26.3 or later, and sits in the class of about 3 microseconds of latency.
MLX's release notes track it as a fast-moving component: a standalone library,
multi-ring refactors, and bandwidth improvements tied to macOS point releases.

The API provides `mx.distributed.init()`, `all_sum`,
`all_gather`, `send` and `recv`, and you launch it with `mlx.launch --hosts`. The
lazy model of Chapter 16 carries over: communication operations are graph nodes,
scheduled and overlapped with the rest of the graph. They are not Metal kernels,
though. In the pinned MLX, every distributed primitive's GPU evaluator throws
"has no GPU implementation", so the collectives run on the CPU stream and the
overlap you get is between that stream and the GPU's.

The sharding strategies are the ones you know. Pipeline parallelism splits the
layers across nodes, with one activation crossing per boundary per token, and
suits decode's serial nature. Tensor parallelism splits the matrix multiplies
across nodes, with collectives per layer; it runs over TCP, but it is only worth
running over the RDMA path. Mixture-of-experts models shard naturally by expert
placement, which is the placement question of Chapter 24.

llama.cpp reached the same place independently: you choose `--split-mode layer`
or `--split-mode tensor` and set the proportions with `--tensor-split`, across
the devices one process can see.

## The performance intuition, from Chapter 22

Which strategy should you pick? Decode is latency dominated, one token at a time, so every message on the
token's critical path adds to the time per token. That is pipeline
parallelism's advantage: one activation crossing per boundary, plus the sampled
token travelling back to the first stage to start the next step. Prefill is
throughput dominated and tolerates schemes with more messages, so tensor
parallelism's per-layer collectives cost a smaller share of the time there than
they would in decode. Fewest messages is not automatically fastest, though.
Pipeline stages run one after another for a single sequence, while a tensor
split lets both machines work on the same layer at once; which wins is a
comparison of complete critical paths, not a count of messages.

The law of Chapter 3 still applies, and it tells you when a split pays. Put
the time to stream a model's weights through one machine at W divided by B,
bytes over bandwidth. A tensor split across p machines streams W divided by pB
per machine and adds communication time T. Splitting wins only when T is less
than the time it saved, W over B times (1 minus 1 over p). With links at tens
of gigabytes per second against local memory in the hundreds, that condition
is hard to meet for a model that fits in one machine, which is why the usual
advice is that a cluster pays off when the alternative is not fitting at all.

"Not fitting" needs care too: Chapter 24's flash-moe runner serves a 209 GB
model from 128 GB of RAM by streaming experts from disk, so one machine can
run such a model, just not with all of it resident. Compare the cluster against
that streaming runner, not against nothing.

For pipeline parallelism the per-token cost is the sum of the stages' weight
streaming times plus the crossings. Take Chapter 23's dense example, about 80
milliseconds per token to stream 39 GB of weights on one machine, and split it
in half: each stage streams about 40 milliseconds. A crossing is the link
latency, about 3 microseconds, plus the activation payload. If the hidden size
is 8,192 values in 16 bits, that is 16 KiB, and at an assumed 40 gigabytes per
second it takes under half a microsecond. Two crossings per token add well
under 10 microseconds against 80 milliseconds of streaming, so for dense
pipeline decode the link is not where the time goes. The assumption to check is
the 40 gigabytes per second: measure your own link before trusting it.

## What transfers from CUDA, and what does not

How much of your multi-GPU CUDA knowledge carries over? The collective
vocabulary transfers whole: an `all_sum` is an all-reduce, an
`all_gather` is an all-gather, and the sharding strategies have the same names
and the same trade-offs as on a multi-GPU NVIDIA box. Two things do not transfer.
The first is the ratio between link bandwidth and local bandwidth. NVLink's
hundreds of gigabytes per second sit in the same class as local memory;
Thunderbolt's tens do not, so schemes that are routine over NVLink, tensor
parallelism for decode in particular, cost more here than they do there. The
second is the first step of scaling: on NVIDIA you can add a card to the
chassis before you add a machine, and here the first step is already another
machine, with its own operating system, its own wired-memory ceiling (Chapter
25), and its own copy of the runtime.

## Setup realities

You need to know three things before you buy cables. RDMA over Thunderbolt must be enabled
once from the Recovery environment. JACCL wants the Thunderbolt 5 mesh wired
directly, with no hub. The order of hosts in the `hostfile` defines the ring.
For the cables, the Recovery step and the host file, follow the public write-up
of clustering two Mac Studios for local inference, listed in the book's sources;
it walks a real build end to end.

## Review the content

1. Two machines each hold half of a model's layers. For one generated token,
   what crosses the link under pipeline parallelism, and why is that a good fit
   for decode?
2. Under what condition does splitting a model that fits in one machine across
   two machines make a token faster, and why is that condition hard to meet
   over Thunderbolt?
3. A trillion-parameter model in 16-bit needs about 2 TB of weights. Why can two
   Mac Studios not hold it in that form, and what does the Kimi K2 proof imply
   about how it was stored?

### Answers

1. One activation crossing at the boundary between the halves, and the sampled
   token going back to the first machine for the next step. Decode is latency
   dominated, so a scheme whose messages are few and small, and whose per-token
   time is dominated by weight streaming rather than by waiting on the link,
   fits it well.
2. When the communication time is less than the streaming time saved: with two
   machines, less than half of the single-machine time W over B. Thunderbolt's
   tens of gigabytes per second against local memory's hundreds make that
   saving small relative to the collectives a tensor split needs per layer.
3. A trillion parameters at 2 bytes each is 2 times 10 to the 12 bytes, 2 TB.
   Two machines at 512 GB hold 1,024 GB, about half. So the weights were
   quantized (Chapter 23). Raw 8-bit codes would take 1,000 GB and leave 24 GB,
   but the affine format of Chapter 23, with a 16-bit scale and bias per group
   of 64, costs 8 plus 32 over 64, which is 8.5 bits per weight, or 1,062.5 GB,
   and does not fit. So either the average width was below 8 bits, or the
   metadata was cheaper than that example; the proof alone does not say which.

With Part 7 behind you, the last part turns to the M5 generation's real matrix
unit in every GPU core and the programming model that reaches it.

Next: [Three matrix engines on one die](28-three-matrix-engines.md)
