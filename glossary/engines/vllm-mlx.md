# vllm-mlx: a vLLM-shaped server on top of mlx-lm

**vllm-mlx is an OpenAI- and Anthropic-compatible inference server for Apple
Silicon written entirely in Python on mlx-lm, mlx-vlm and mlx-audio
(Apache-2.0, waybarrios/vllm-mlx, pinned here at commit 078e94e9, version
0.5.0, about 72,900 lines). It has no Metal kernels of its own and no
`mx.fast.metal_kernel` calls; every kernel that runs is stock MLX. It is in
this glossary for two reasons: it has a paper, arXiv 2601.19139, whose
numbers you can hold against the repository, and its "paged" and "quantized"
caches are the clearest worked example of why those words mean nothing until
the attention kernel reads the result.**

Fetched by `fetch.sh` as `code/vllm-mlx`. Line references are into that
pinned copy; the paper is external.

## What the speed is

The largest documented effect is prefix reuse, and it only helps time to first
token. The default cache keys whole token tuples, keeps them in a sorted list
so a prefix sorts before every extension of it, and finds the longest common
prefix by bisection (`vllm_mlx/memory_cache.py:1036-1078`). The warm-prompt
table for Qwen3-4B at concurrency 1 goes from 4,937 ms to 2,191 ms, 2.25x,
with all twelve configurations between 1.30x and 2.25x on an M4 Max; decode is
"neutral (within ±5%)" for dense models and the MoE model drops 20 to 35
percent at concurrency 4 and above (`docs/guides/warm-prompts.md:90-118`).

Continuous batching is delegated outright: "mlx-lm's BatchGenerator already
implements continuous batching at the token level, so we use it as the
backend" (`vllm_mlx/scheduler.py:1346-1347`). The gain is the
[decode](../techniques/decode-vs-prefill.md) arithmetic: at batch one every
weight byte is streamed for one token, at batch B for B tokens. The measured
table is Qwen3-0.6B at 8-bit from 328.1 to 1,111.8 tok/s with five requests,
3.39x, and Llama-3.2-3B at 1.51x (`docs/benchmarks/llm.md:30-38`). A
300-line patch chunks long prompts so a generation step runs between chunks
(`scheduler.py:203-217`), which is the Metal-cannot-preempt rule from the
[command-buffers](../metal/command-buffers.md) page; with the pinned mlx-lm the
patch's precondition no longer holds and the native chunking is used instead
(`scheduler.py:745-788`). Lowering a mixture-of-experts model's active experts
from 8 to 4 gives 126.5 to 147.3 tok/s, +16.5 percent, by streaming half the
expert bytes; the quality table is still "TBD" (`docs/guides/moe-top-k.md:41-59`).

## Paged in name, quantized at rest

`PagedCacheManager` ports vLLM's block pool faithfully: 64-token blocks, a
doubly linked free list, reference counts, copy-on-write, SHA-256 chain hashes
(`vllm_mlx/paged_cache.py:1-75`). But attention never reads blocks. On a hit,
the cache reconstruction slices the stored per-block tensors and concatenates them
per layer back into one contiguous `KVCache` (`vllm_mlx/prefix_cache.py:753-768`),
so every hit copies the whole prefix's K and V once before the first token and
the decode kernel is unchanged. The project's own table shows the result:
"1.00x" and "0.99x" on an M1 Max, "1.12x" on an M4 Max, and the feature is
off by default and labelled experimental (`docs/benchmarks/llm.md:114-177`;
`scheduler.py:107-110`). Compare the [kv-cache](../techniques/kv-cache.md)
page's arithmetic: paging pays on a GPU only when the attention kernel gathers
pages itself, which is what [vllm-metal](vllm-metal.md) builds.

KV-cache quantization has the same shape. With the flag set, entries of at
least 256 tokens pass through `mx.quantize` when stored and `mx.dequantize`
when fetched, rebuilding a plain cache: "the model always receives normal
KVCache with plain arrays" (`vllm_mlx/mllm_scheduler.py:306-308`). Residency
drops; decode bytes per token do not, and a dequantize dispatch is added per
layer per hit. The comment's "~4x (BF16→Q8)" works out to about 1.9x once the
scale and bias per group of 64 are counted (inferred arithmetic). This is the
[quantization](../mlx/quantization.md) page's distinction between a storage
format and a kernel that consumes it, drawn by a project that has only the
first.

## Where Metal shows through

With no kernels, the GPU-level engineering is stream and memory hygiene. One
generation stream is bound per owning thread, because "MLX streams exist only
in the thread that created them" (`vllm_mlx/mlx_streams.py:28-35`);
`mx.clear_cache()` runs on an adaptive interval starting at every 32 steps "to
free Metal command buffers" (`scheduler.py:1451-1454`). The deferred-removal
comment records a real failure: two threads submitting on one stream assert
"encodeSignalEvent with uncommitted encoder"
(`vllm_mlx/mllm_batch_generator.py:670-677`). The SSD tier has to materialize
the MLX-to-numpy copy on the thread that owns the GPU stream, because doing it
on the writer thread "aborts the process" (`vllm_mlx/ssd_cache.py:947-951`),
and because numpy cannot take bf16, KV makes a bf16 to fp16 round trip on disk
(`ssd_cache.py:1006-1014`). Multi-token-prediction verification is greedy
argmax with one host sync per step, and acceptance is batch-wide
(`scheduler.py:1103-1107`): one rejecting row rolls back every row, so the gain
shrinks as the batch grows. The optimistic flag removes the sync by skipping
verification (`scheduler.py:1075-1076`), which shows what the wait costs only
by giving up correctness.

The negative result worth keeping: constructing a streaming detokenizer costs
"52.7ms each", turning 681 tok/s into 275, "-60% slower"
(`docs/benchmarks/llm.md:211-228`); a 2x algorithmic win erased by object
construction, the [cheap tricks](../war-stories/cheap-tricks.md) page's
category of loss.

## The paper against the repository

The abstract claims 21 to 87 percent over llama.cpp, 4.3x aggregate throughput
at 16 concurrent requests, up to 525 tok/s on an M4 Max, and 28x on repeated
image queries (external). None of the four is reproduced in the repository.
There is no llama.cpp comparison script or result; `benchmarks/` holds two
unrelated files, and the paper's own Table 1 gives Qwen3-30B-A3B 1.17x, under
its 21 percent floor. The 525 tok/s is Qwen3-0.6B at 4-bit in the paper; the
repository documents the 8-bit model at 402.3 and 417.9 tok/s
(`docs/benchmarks/llm.md:13`, `README.md:111`). The batching table stops at
five requests and 3.39x, and the paper's body says about 3.7x where its
abstract says 4.3x. The 28x is per-turn latency for a vision model; the
repository's image-cache "results" are lookup times of 0.10 to 0.24 ms
(`docs/benchmarks/image.md:83-115`), and the committed
`vlm_benchmark_results.json` labels its image "Yellow Labrador" while every
answer says "polar bear" at a flat 0.68 s from 224 by 224 to 1080p. The paper
says the image cache hashes decoded pixels; the code hashes the file bytes,
the URL or the base64 string, truncated to 16 hex characters
(`vllm_mlx/mllm_cache.py:149-156`). None of this makes the server slow. It
makes it the worked example for [question three's](../war-stories/three-questions.md)
discipline: measured where, against what, with which script.

Next: [vllm-metal](vllm-metal.md)
