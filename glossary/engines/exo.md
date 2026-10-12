# exo: a cluster that writes no kernels

**exo joins several Macs into one inference cluster and places a model across
them (Apache-2.0, exo-explore/exo, pinned here at commit 21a54c5e, version
0.3.70). It is about 32,000 lines of Python on MLX and mlx-lm, 1,200 lines of
Rust for discovery and messaging, and zero Metal source files. Every GPU
dispatch is MLX's own. Its speed is the one lever this glossary's single-box
pages cannot pull: dividing the bytes each machine streams per token by the
number of machines, and paying for it with collectives whose latency, not
bandwidth, sets the ceiling.**

Fetched by `fetch.sh` as `code/exo`. Line references are into that pinned
copy. Both MLX and mlx-lm are personal forks (`pyproject.toml:84, 96`); the MLX
branch is named `address-rdma-gpu-locks`.

## What each node runs

Each node runs mlx-lm's model code and generation loop. exo replaces only the
layers at shard boundaries and the linear layers it splits, then calls mlx-lm's
`stream_generate` or wraps its `BatchGenerator` with `prefill_step_size=4096`
(`src/exo/worker/engines/mlx/generator/batch_generate.py:105-108`). A one-Mac
exo is rewritten to pipeline sharding on the ring backend
(`src/exo/master/placement.py:245-252`), which is mlx-lm plus a server. The
project is entirely a question of [where the bytes go](../techniques/kv-cache.md)
and [how the machines wait on each other](../metal/synchronization.md).

## Tensor parallelism: bytes per node divided by N

Decode on one Mac is [bandwidth bound](../techniques/decode-vs-prefill.md).
Splitting every layer's weights N ways means each node streams 1/N of the
weights per token. `tensor_auto_parallel` builds MLX's `shard_linear` in two
modes, column split and row split followed by an all-reduce
(`src/exo/worker/engines/mlx/auto_parallel.py:456-468`); for Llama the q, k, v,
gate and up projections are column split and the o and down projections are
row split, with head counts divided by N (`auto_parallel.py:628-652`). That is
two all-reduces per layer per token, and ten architecture-specific strategies
follow (`auto_parallel.py:605-1593`). Mixture-of-experts layers are not
expert-parallel: every expert's gate, up and down are split across all ranks
and the block output is all-summed (`auto_parallel.py:729-768`). The trade is
one extra collective per layer for identical routing on every node and no load
imbalance. One comment records the opposite trade: DeepSeek V4's `wo_b` is
replicated because "61 extra all_gathers/token cost more than running wo_b on
every rank" at about 30M FLOPs per layer (`auto_parallel.py:803-824`).

## RDMA over Thunderbolt 5, as actually turned on

exo does not implement RDMA. It detects it, builds a device matrix, and hands
MLX three environment variables. Every 10 seconds each node runs `rdma_ctl
status` and `system_profiler SPThunderboltDataType`
(`src/exo/utils/info_gatherer/info_gatherer.py:214-233`); each Thunderbolt
receptacle is mapped to a BSD interface and named `rdma_<iface>`. A link
counts as an RDMA edge only when both ends report enabled
(`src/exo/shared/apply.py:412-438`). The master keeps only placements where
every edge is RDMA, and the JACCL backend needs a full mesh: a missing pair
raises "Current jaccl backend requires all-to-all RDMA connections"
(`src/exo/master/placement_utils.py:296-325`). The worker then writes the
matrix to a file and sets `MLX_IBV_DEVICES`, `MLX_RANK` and
`MLX_JACCL_COORDINATOR` before `mx.distributed.init(backend="jaccl")`
(`src/exo/worker/engines/mlx/utils_mlx.py:122-143`). Without RDMA it falls
back to MLX's ring backend over TCP. The setup rules are in the README: RDMA
enabled from Recovery, Thunderbolt 5 cables, a full mesh, and macOS versions
that "must match exactly (even beta version numbers)" (`README.md:301-307`).

## Pipeline parallelism fits models; it does not speed decode

Layers are split in proportion to each node's available RAM
(`placement_utils.py:60-75`). At each boundary the first layer receives from
rank r-1 and the last layer sends to rank r+1; in decode every stage then
all-gathers the output (`auto_parallel.py:120-195`), so a decode token is
strictly sequential across stages and streams the same total bytes as one
machine would, just in turn. The speed work is in prefill:
`pipeline_parallel_prefill` splits prompts of at least 4,096 tokens into chunks
of `4096 // min(4, world_size)` and staggers the ranks with dummy iterations so
rank 1 processes chunk k while rank 0 processes chunk k+1; the docstring at
`generate.py:164-196` draws the timeline. Pipeline boundaries also carry a
Metal-specific cost: before a `recv`, the input is evaluated and the receive is
evaluated alone "to avoid GPU timeout errors", keeping the distributed
operation on the CPU, which has no timeout (`auto_parallel.py:134-138`). Every
layer boundary is therefore two host synchronizations, a correctness
constraint paid in [waits per token](../metal/command-buffers.md).

## What the numbers are, and what they are not

The README claims "up to 1.8x speedup on 2 devices and 3.2x speedup on 4
devices" and a "99% reduction in latency between devices" (`README.md:25-27`).
Neither has a results file. The third-party numbers are three chart images
from Jeff Geerling's four-Mac-Studio build, linked from `README.md:41-65`:
decode tokens per second, llama.cpp over TCP against exo over RDMA, at 4 nodes
15.2 vs 31.9 for Qwen3 235B, 14.6 vs 32.5 for DeepSeek V3.1 671B, and 16.4 vs
28.3 for Kimi K2. Read against the same charts' 1-node bars, exo's 4-node gain
is about 1.6x, not 3.2x, and llama.cpp gets slower as nodes are added while exo
gets faster. That pattern is the lesson: at batch one the ceiling is set by the
latency of two collectives per layer, not by link bandwidth, which is why the
[distributed](../mlx/distributed.md) page says the transport decides. What the
repository does provide is the tool to re-measure the exo side,
`bench/exo_bench.py` with `--sharding tensor`, with a written methodology
(`bench/METHODOLOGY.md:27-65`): end-of-sequence banned, prefix cache off,
server-side timing.

## What else is there

A prefix cache keeps prior prompt caches, returns partial hits, snapshots
non-trimmable recurrent state for hybrid models, and evicts above a memory
threshold of 0.70 to 0.85 depending on RAM
(`src/exo/worker/engines/mlx/cache.py:33-48, 232-370`); under tensor
parallelism the pressure check takes the maximum across ranks. A decode
instance can fetch its KV cache from a separate prefill instance for prompts
over 1,000 tokens (`generate.py:640-662`), and the backend list includes CUDA,
so the intended shape is compute-heavy prefill on an NVIDIA box and
bandwidth-heavy decode on Macs. No speculative decoding, no offload, no
quantization format of its own. Numerics: a test asserts tensor-parallel logits
are bit-exact against one device (`test_tp_bit_exact.py:376-431`), though it
calls `tensor_auto_parallel` with a signature the current code no longer has.

Read next to [oMLX](omlx.md): where oMLX fills a ledger of kernels and
scheduler rules, exo gets everything from topology and the interconnect, and
its honest numbers are a quarter of its advertised ones.

Next: [vllm-mlx](vllm-mlx.md)
