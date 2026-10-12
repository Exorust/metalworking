# Rapid-MLX: speculative decoding priced by the matmul tile

**Rapid-MLX is an OpenAI- and Anthropic-compatible inference server for Apple
Silicon aimed at coding agents (Apache-2.0, raullenchai/Rapid-MLX, pinned here
at commit 6364ee8a), written in Python on stock MLX and mlx-lm, about 371,600
lines, with one `.metal` file and about twenty modules of inline
`mx.fast.metal_kernel` source. Its headline is 4x over `mlx_lm.generate` on a
whole-file code edit, 46.4 to 198.4 tok/s, from speculative decoding. What
makes it worth reading is that it prices every speculative choice against one
hardware fact: MLX's quantized matmul handles rows in 32-row tiles, and row 33
costs 89 percent more than row 32.**

Fetched by `fetch.sh` as `code/rapid-mlx`. Line references are into that
pinned copy.

## The tile as a cost function

Speculative decoding verifies K drafted tokens in one (K+1)-row forward. In a
code comment, the authors measured that forward on Qwen3.8-27B at 4-bit: 32
rows cost 342.26 ms and 33 rows cost 647.52 ms, +89 percent, and the same step
appears in the kernel alone on an M3 Ultra and an M4 Pro (17408 by 5120, 4-bit,
group 64: 655.9 to 1034.9 us and 1442.8 to 2567.2 us)
(`rapid_mlx/spec_decode/mtp/prompt_lookup.py:164-173`). Below the edge, rows
are not free either, because the vector path runs up to the GEMM switch: 66.8
ms at one row, 69.7 at two, 262.5 at nine, then flat within about 3 percent
from 13 to 32 rows (`prompt_lookup.py:254-273`). So copy drafts are capped at
31 tokens, widths that would cross a tile are rounded to the tile edge
(`prompt_lookup.py:189-190, 393-403`), prefill chunks under contention are
rounded down to whole 32-row tiles because "a partial tile costs a full one"
(`rapid_mlx/scheduler.py:6743-6783`), and message-boundary cache snapshots are
tile-aligned because a 33-token prefill costs 640 ms against 336 for 32
(`scheduler.py:9122-9135`). The [tiling](../techniques/tiling.md) page
describes the tile from inside the kernel; this project meets it from outside,
as a step function in the bill.

## Two drafters and the gates that keep prose at parity

The multi-token-prediction path is vendored from the still-open mlx-lm pull
request 990 (`rapid_mlx/spec_decode/mtp/generator.py:1-35`): Qwen ships one
extra decoder layer that predicts token t+2 from the hidden state at t, so
this is chain drafting, one token deep per head. On top sits prompt lookup, an
n-gram index over the prompt only; generated text is used as the query and
never added to the index, so the model cannot copy its own repetition into a
fast loop (`prompt_lookup.py:71-76`). Every draft is verified in one forward
with a single host sync, and `mx.async_eval` launches the next draft while the
host detokenizes (`generator.py:1240-1263, 1686`), the placement the
[lazy evaluation](../mlx/lazy-evaluation.md) page recommends.

Two controllers decide whether speculation runs at all. The depth controller,
ported from Ollama, picks the depth that maximizes expected committed tokens
per expected cost, with depth zero meaning plain decode
(`spec_decode/mtp/draft_k_controller_v2.py:1-27`). A comment records the
failure that shaped it: one first-touch cost sample of about 22 ms against a
steady 16 ms permanently flipped the decision, so prose "spent 3/4 of its
budget paying drafter cost for near-zero accept"; the fix seeds four samples
per depth (`draft_k_controller_v2.py:84-101`). Copy drafts get a per-request
gate that compares tokens per millisecond over a 32-round window and sizes the
copy block like a TCP congestion window: a 2-row probe, doubling while blocks
come back fully accepted, dropping to the acceptance average plus headroom
once one comes back short (`prompt_lookup.py:406-431, 589-619`). The reason
it must be per request is measured: on Qwen3.6-27B on an M3 Ultra copy drafts
accepted about 4 rows and cost 2.5 to 4.1 percent of throughput, while on
Qwen3.8-27B on an M4 Pro they accepted 10 to 14 and gained 28 to 33 percent
(`prompt_lookup.py:413-418`). The code-comment table for the 27B at 4-bit:
rename +51 percent and bugfix +39 percent tokens per round, prose within +0.9
to -0.7 (`spec_decode/mtp/qwen3_5_inject.py:973-981`).

## Rollback for a state you cannot trim

A rejected draft has to roll the cache back. Attention layers trim their
[KV cache](../techniques/kv-cache.md); a gated delta-net recurrent state is a
running summary and cannot be trimmed (`generator.py:757-813`). Rapid patches
`GatedDeltaNet.__call__` so a verify of more than two positions calls one
kernel that runs the recurrence once and writes the (conv, state) pair at
every interior boundary as a restore point
(`spec_decode/mtp/cache_patch.py:284-331`). The kernel keeps each lane's slice
of the state in registers (`float state[n_per_t]`), reduces with `simd_sum`,
uses no threadgroup memory and no barriers, and writes the state after each of
the first T-1 steps (`rapid_mlx/kernels/qwen4_gdn_verify.py:28-112`). The
trade is bytes for replay: one fp32 state per boundary per head written, no
re-running of the accepted prefix. Documented on an M3 Ultra, Qwen3.8-27B at
fixed depth 3: 45.87 to 52.28 tok/s, +14.0 percent, identical 256-token hashes
across runs, with the exact reproduction command
(`docs/engineering/performance/2026-09-07-qwen38-mtp-gdn-verify.md:17-66`).
The prefix cache uses the same immutability: recurrent layers are checkpointed
every 2,048 tokens by keeping a reference to the superseded MLX array, so
nothing is copied; editing the last of 65 sections of a 5,700-token document
drops time to first token from 19.5 s to 7.5 s, and recording costs 4 percent
(`rapid_mlx/hybrid_state_checkpoints.py:17-60`;
`docs/engineering/performance/2026-09-13-hybrid-cache-checkpoints.md:97-131`).

## Where speculation loses

The project documents its own loss. Batched verify at 2 to 4 lanes costs too
much: 12 rows cost 106.5 ms against 23.3 ms for one row on the 9B, so a 4-lane
verify costs about 2.8 to 4.6 single steps for about 2.4 tokens per lane
(`docs/engineering/performance/2026-10-02-mtp-concurrency-m4-pro.md:18-39`).
The shipped policy runs speculation for a lone request and plain batched
decode beyond that, which takes Qwen3.6-35B at four concurrent streams from
109.6 to 164.3 tok/s, against 166.1 with speculation off (`:64-73`). The
decode-side fusions are the same family as [oMLX](omlx.md)'s and each has a
benchmark script and a token-identity check: the fused gated delta-net decode
step 114.65 to 128.00 tok/s, +11.7 percent; the fused router tail +5.1; whole
step `mx.compile` replay over fixed KV buckets 124.55 to 143.38, +15.1, batch
one only (`docs/engineering/performance/2026-09-12-qwen36-35b-fused-gdn-decode.md:71-79`,
`2026-09-13-qwen36-compiled-decode.md:36-45`). One fusion was dropped after a
one-ULP reassociation changed greedy tokens (`2026-09-12-...-fused-gdn-decode.md:90-100`).

## Caveats

The two README headlines, 4x over mlx-lm and 3.0x over Ollama, have their raw
data on external pages; the 4x is a best case on one task, the median over
18 tasks is 1.50x, a long-prompt agent turn is 1.05x (`README.md:68-83`). The
authors state that a multi-row verify "does not reproduce a one-row step's
logits bit for bit", which made two identical greedy requests return different
text; a separate reproducible schedule exists for that reason
(`spec_decode/mtp/reproducible_depth.py:1-15`). The "radix" cache is a lookup
index over whole per-entry caches, not shared blocks, and its own docstring
says it "only indexes token sequences, not their KV state"
(`rapid_mlx/runtime/radix_index.py:5-11`). Live KV is bf16; int4 and int8 are
opt-in and documented as 27 and 36 percent slower at 16K from dequantizing on
read (`rapid_mlx/cli_parser.py:899-907`). The one place Metal 4 tensor ops
appear, the off-by-default lane matmul, is used not for peak throughput but so
that a row gets the same bits alone or inside a verify block
(`rapid_mlx/kernels/lane_matmul/matmul.py:16-27`), with a `simdgroup_matrix`
twin for M1 to M4.

That ends the engines section and the glossary. [SOURCES.md](../../SOURCES.md) has the
full annotated list of repos, writeups, and people worth following to stay current.
