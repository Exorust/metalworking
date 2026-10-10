# Scaling out across machines

## Where we are

Chapter 6 said there is one GPU per Apple Silicon machine, no NVLink and no
second card, and Chapter 3 said the way to scale inside a machine is to buy more
unified memory. This chapter is the other direction: when a model does not fit
in one machine, `mx.distributed` runs it across several, and with macOS 26.3 and
Thunderbolt 5 the connection between them is remote direct memory access at
microsecond latencies, which is why "no NVLink" stopped being the end of the
scaling story. The chapter is short, because the performance intuition is the
one Chapter 22 already built.

## What it is, against NCCL

The CUDA counterpart is NCCL plus the fabric underneath it. The comparison in
numbers: NVLink moves hundreds of gigabytes per second between GPUs in one
chassis; Thunderbolt 5 RDMA moves tens of gigabytes per second between chassis.
What makes the Apple version interesting anyway is the memory arithmetic. Each
node brings up to 512 GB of unified memory, so two Mac Studios hold a
trillion-parameter model that no single consumer NVIDIA machine can load at any
speed. The public existence proof is Kimi K2, a trillion-parameter model,
running across two M3 Ultra machines through `mx.distributed`.

The pieces, from the MLX distributed documentation and session 233 of WWDC 2026:

- **Backends.** `ring` runs over TCP on any network, works everywhere, and is
  slow. JACCL does RDMA over Thunderbolt 5, needs macOS 26.3 or later, and sits
  in the class of about 3 microseconds of latency. MLX's release notes track it
  as a fast-moving component: a standalone library, multi-ring refactors, and
  bandwidth improvements tied to macOS point releases.
- **The API is collective-shaped.** `mx.distributed.init()`, `all_sum`,
  `all_gather`, `send` and `recv`, launched with `mlx.launch --hosts`. The lazy
  model of Chapter 16 carries over: communication operations are graph nodes,
  batched and overlapped like everything else.
- **Sharding strategies.** Pipeline parallelism splits the layers across nodes,
  with one link crossing per token per boundary, and suits decode's serial
  nature. Tensor parallelism splits every matrix multiply, with collectives per
  layer, and demands the RDMA path. Mixture-of-experts models shard naturally by
  expert placement, which is the question-3 territory of Chapter 24. llama.cpp
  reached the same place independently; its Metal backend grew tensor-split
  support.

## The performance intuition, from Chapter 22

Decode is latency dominated, one token at a time, so what matters is the number
of round trips per token, which is pipeline parallelism's advantage: one crossing
per boundary per token. Prefill is throughput dominated and tolerates chattier
schemes, so tensor parallelism's per-layer collectives cost less there than they
would in decode.

And the law of Chapter 3 still rules. A cluster whose links are slower than its
local memory bandwidth, tens of gigabytes per second against hundreds, only wins
when the alternative is not fitting at all. That is exactly the regime that
local language-model clusters live in: the model is too large for one machine's
memory, so the question is not whether the cluster is faster than one machine
running the model, because one machine cannot, but whether the cluster's
tokens per second are worth having at all. By the arithmetic of Chapter 22, a
pipeline-parallel decode step pays the per-machine bandwidth time for its share
of the weights plus one link crossing per boundary, and with crossings of a few
microseconds against weight streaming of tens of milliseconds, the link is not
where the time goes.

## What transfers from CUDA, and what does not

The collective vocabulary transfers whole: an `all_sum` is an all-reduce, an
`all_gather` is an all-gather, and the sharding strategies have the same names
and the same trade-offs as on a multi-GPU NVIDIA box. Two things do not transfer.
The first is the ratio between link bandwidth and local bandwidth, which is far
worse here, so schemes that are routine on NVLink, tensor parallelism for decode
in particular, cost more than their CUDA reputation suggests. The second is the
unit of scaling: on NVIDIA you add cards to a chassis, and here you add
machines, each with its own operating system, its own wired-memory ceiling,
Chapter 25, and its own copy of the runtime.

## Setup realities

Three things to know before buying cables. RDMA over Thunderbolt must be enabled
once from the Recovery environment. JACCL wants the Thunderbolt 5 mesh wired
directly, with no hub. And the order of hosts in the `hostfile` defines the ring.
A public write-up of clustering two Mac Studios for local inference walks a real
build end to end, cables, Recovery step and host file included, and it is listed
in the book's sources.

## Check yourself

1. Two machines each hold half of a model's layers. For one generated token,
   how many link crossings does pipeline parallelism cost, and why is that the
   right choice for decode?
2. Why does the chapter say a Thunderbolt cluster "only wins when the
   alternative is not fitting at all"?
3. A trillion-parameter model in 16-bit needs about 2 TB of weights. Why can two
   Mac Studios not hold it in that form, and what does the Kimi K2 proof imply
   about how it was stored?

### Answers

1. One, at the boundary between the two halves, per token. Decode is latency
   dominated, so the scheme with the fewest round trips per token wins.
2. Because the link, at tens of gigabytes per second, is slower than local memory
   bandwidth, so any work that could have stayed on one machine is slower when
   split; the cluster pays off only for models that one machine cannot load.
3. Two machines at 512 GB each hold 1 TB, half of 2 TB, so the weights must have
   been quantized, Chapter 23, to 8 bits or fewer to fit.

Part 7 is done. The last part is about the hardware that arrived after most of
this book's kernels were written: a real matrix unit in every GPU core, and the
programming model that reaches it.

Next: [Three matrix engines on one die](28-three-matrix-engines.md)
