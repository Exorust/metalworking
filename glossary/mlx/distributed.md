# Distributed MLX

**`mx.distributed` runs one model across several Macs. With macOS 26.3+ and
Thunderbolt 5, the JACCL backend does RDMA between machines at microsecond
latencies, which is why "no NVLink" stopped being the end of the scaling
story.**

CUDA equivalent: NCCL plus the fabric underneath it. The honest comparison:
NVLink moves hundreds of GB/s between GPUs in one chassis; Thunderbolt 5
RDMA moves tens of GB/s between chassis. What makes the Apple version
interesting anyway is the memory math: each node brings up to 512 GB of
[unified memory](../machine/unified-memory.md), so two Mac Studios hold a
1T-parameter model that no single consumer NVIDIA box can load at any speed.
The public existence proof is Kimi K2 (1T parameters) running across two
M3 Ultras via `mx.distributed`.

The pieces
([MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html),
[WWDC26 233](https://developer.apple.com/videos/play/wwdc2026/233/)):

- **Backends.** `ring` (TCP over any network, works everywhere, slow) and
  **JACCL** (RDMA over Thunderbolt 5, macOS 26.3+, ~3 µs latency class). MLX
  release notes track it as a fast-moving component: standalone library,
  multi-ring refactors, bandwidth improvements tied to macOS point releases.
- **The API is collective-shaped**: `mx.distributed.init()`, `all_sum`,
  `all_gather`, `send`/`recv`, launched with `mlx.launch --hosts`. The
  [lazy-evaluation](lazy-evaluation.md) model carries over: communication ops
  are graph nodes, batched and overlapped like everything else.
- **Sharding strategies.** Pipeline (layers split across nodes; one link
  crossing per token per boundary) suits decode's serial nature; tensor
  parallelism (every matmul split; collectives per layer) demands the RDMA
  path; MoE experts shard naturally by placement
  ([three-questions territory](../war-stories/three-questions.md)). llama.cpp
  reached the same place independently: its Metal backend grew tensor-split
  support.

The performance intuition transfers from
[decode-vs-prefill](../techniques/decode-vs-prefill.md): decode is latency
dominated, so what matters is the per-token round-trip count (pipeline
parallelism's advantage); prefill is throughput dominated and tolerates
chattier schemes. And the platform's [law](../machine/unified-memory.md)
still rules: a cluster whose links are slower than local memory bandwidth
only wins when the alternative is not fitting at all. That is exactly the
regime local-LLM clusters live in.

Setup realities worth knowing before buying cables: RDMA over Thunderbolt
requires enabling it once from Recovery mode; JACCL wants the TB5 mesh wired
directly (no hub); and `hostfile` order defines the ring. The
[clustering writeup](https://www.sean-weldon.com/blog/2026-06-16-clustering-mac-studios-for-local-ai-apple-rdma-over-thunderbolt-5)
walks a real two-Studio build end to end.

Next: [Techniques](../techniques/arithmetic-intensity.md)
