# The KV Cache

**The KV cache stores every past token's key and value vectors so attention
never recomputes them. It is the thing [decode](decode-vs-prefill.md) streams
in full on every token, which makes it the platform's most consequential data
structure.**

CUDA equivalent: the same structure everywhere (vLLM's paged KV made it
famous); the platform inflection is that on
[bandwidth-bound](../machine/unified-memory.md) hardware its size converts
directly into decode latency.

The arithmetic that runs everything: per token, per layer, the cache holds one
K and one V vector per KV head. For a typical dense 7-8B model
(`n_layers ≈ 32`, `n_kv_heads × head_dim ≈ 1024`, fp16) that is ~128 KB per
token, so a 32K-token context carries a ~4 GB cache, and every decoded token
streams all of it through the [attention kernels](flash-attention.md). Three
consequences the glossary keeps meeting:

- **Long context turns decode into a cache-streaming benchmark.** The
  [sparse-V war story](../war-stories/sparse-v.md)'s +22.8% at 32K is
  precisely a KV-cache traffic deletion.
- **GQA is a KV-cache compression scheme** as much as an attention variant:
  fewer KV heads = a smaller cache = faster decode, which is why
  [steel attention](../kernels/steel-attention.md) implements it as a stride
  trick (`kv_head_idx = tid.y / gqa_factor`).
- **Quantizing the cache is the next lever after quantizing weights**, and it
  is asymmetric in a way weight quantization is not.

**The K/V asymmetry.** The
[turboquant-mlx project](https://github.com/arozanov/turboquant-mlx/blob/6e928d715595dee9f6b6cc3968baa44e1f408d28/README.md)
measured that quantizing K to 4 bits or below breaks greedy decode (softmax is
sensitive to small score perturbations, and K sits upstream of the softmax),
while V tolerates 3-bit quantization fine (V is consumed by a weighted
average, which absorbs noise). Its mixed K8/V4 configuration cut a 6.21 GB
fp16 cache to 5.08 GB with greedy output verified identical to baseline. The
general rule: treat K's precision as accuracy-critical and V's as a bandwidth
knob. Practitioner reports of `kv_bits=4` corrupting long prefilled contexts
are this asymmetry being ignored, and note the flip side: quantized KV does
not speed up [prefill](decode-vs-prefill.md) at all, since prefill's cost is
the matmuls, not cache streaming.

**Layout matters at the kernel level.** The simple layout is one contiguous
ring per layer; serving stacks use *paged* layouts (fixed-size blocks plus an
indirection table) so many sequences share memory without fragmentation. That
world exists on Metal: the official vLLM plugin ships a paged varlen
attention kernel, and MTPLX carries a
[paged-attention Metal kernel family](https://github.com/youssofal/MTPLX/tree/ed1c8eea501689b744c13bec6a99ee2d36d26ab5/vllm_metal/metal/kernels_v2)
(`pagedattention.metal`, `reshape_and_cache.metal`, `gather_kv_cache.metal`).
Serving-grade batching is its own topic; for single-user decode, the
contiguous cache plus the [split-KV decode
kernels](../techniques/decode-vs-prefill.md) is the whole story.

Next: [Decode vs prefill](decode-vs-prefill.md)
