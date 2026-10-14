# Distributed MLX

**`mx.distributed` runs one model across several Macs. With macOS 26.2+ and
Thunderbolt 5, the JACCL backend does RDMA between machines at microsecond
latencies, which is why "no NVLink" stopped being the end of the scaling
story.**

CUDA equivalent: NCCL plus the fabric underneath it. The comparison:
NVLink moves hundreds of GB/s between GPUs in one chassis; Thunderbolt 5
RDMA moves tens of GB/s between chassis. What makes the Apple version
interesting anyway is the memory math: each node brings up to 512 GB of
[unified memory](../machine/unified-memory.md), so two Mac Studios hold a
1T-parameter model that no single consumer NVIDIA box can load at any speed.
The public existence proof is Kimi K2 (1T parameters) running across two
M3 Ultras via `mx.distributed`; Apple's own demo in WWDC26 233 runs Kimi 2.6
across four, because at 8 bits the weights alone are about a terabyte.

The pieces
([MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html),
[WWDC26 233](https://developer.apple.com/videos/play/wwdc2026/233/)):

- **Backends.** `ring` (TCP over any network, works everywhere, slow) and
  **JACCL** (RDMA over Thunderbolt 5, macOS 26.2+, ~3 µs latency class), which
  itself runs in two topologies: a full mesh (`--backend jaccl`, lowest
  latency) or a ring (`--backend jaccl-ring`, fewer cables, two or three per
  neighbour for bandwidth), and on a wired mesh picks mesh or ring routing per
  operation by message size. MLX release notes track it as a fast-moving
  component: standalone library, multi-ring refactors, bandwidth improvements
  tied to macOS point releases.
- **The API is collective-shaped**: `mx.distributed.init()`, `all_sum`,
  `all_gather`, `send`/`recv`, launched with `mlx.launch --hosts` or
  `--hostfile`; `mlx.distributed_config` probes the Thunderbolt ports, writes
  the JSON hostfile (per node: `ssh` name, coordination `ips`, per-peer `rdma`
  device names) and with `--auto-setup` configures each link for RDMA. Apple's
  setup also exports `MLX_METAL_FAST_SYNCH=1`, since compute runs on the GPU
  and communication on the CPU. The
  [lazy-evaluation](lazy-evaluation.md) model carries over: communication ops
  are graph nodes, batched and overlapped like everything else.
- **Sharding strategies.** Pipeline (layers split across nodes; one link
  crossing per token per boundary) is the capacity split: it lets a model that
  does not fit run, but Apple notes it "does not speed up the inference"
  because each token still visits the stages in order. Tensor parallelism
  (every matmul split; collectives per layer) is the speed split and the MLX LM
  default (`--pipeline` selects the other); it demands the RDMA path. MoE
  experts shard naturally by placement
  ([three-questions territory](../war-stories/three-questions.md)). llama.cpp
  reached the same place independently: its Metal backend grew tensor-split
  support.

The performance intuition transfers from
[decode-vs-prefill](../techniques/decode-vs-prefill.md): decode is latency
dominated, so what matters is the per-token collective count and each
collective's latency, not link bandwidth, because a decode all-reduce carries
only a hidden-size vector; prefill is throughput dominated and tolerates
chattier schemes. The platform's [law](../machine/unified-memory.md) sets the
rest: over TCP, a cluster only wins when the alternative is not fitting at all;
over the RDMA mesh, tensor-parallel decode can also beat one machine on a model
that fits, and Apple shows a 27B model at nearly 3× on four M3 Ultras
(WWDC26 233).

Setup realities worth knowing before buying cables: RDMA over Thunderbolt
must be enabled once per machine and rebooted (a System Settings toggle,
"Enable RDMA over Thunderbolt", in the WWDC26 session; `rdma_ctl enable` from
the Recovery environment's Terminal on macOS 26 builds without that pane);
JACCL wants the TB5 links wired directly between machines (no hub); and
`hostfile` order defines the ring. The
[clustering writeup](https://www.sean-weldon.com/blog/2026-06-16-clustering-mac-studios-for-local-ai-apple-rdma-over-thunderbolt-5)
walks a real two-Studio build end to end.

Next: [Techniques](../techniques/arithmetic-intensity.md)
