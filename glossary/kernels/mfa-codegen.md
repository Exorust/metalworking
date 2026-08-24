# Case Study: metal-flash-attention

**[philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
(MIT): 83% ALU utilization on M1 Max, and a structural bet visible in one fact.
There are no `.metal` files in the repo. Every kernel is assembled as a Swift
string for the exact problem shape, then
[compiled at runtime](../metal/compilation-pipeline.md).**

The assembly point, MSL as string interpolation:

```swift
    return """
    \(createMetalSimdgroupEvent())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;

    \(createConstants())

    // Declare the function.
    kernel void attention(
      \(createBufferBindings())
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
      ...
      \(createSetup())
      \(createLoop())
      \(createCleanup(type: type))
    }
    """
```
— [`AttentionKernel+Source.swift:23-54`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Source.swift#L23-L54), abridged

Where [MLX templates](steel-gemm-fused.md) and
[llama.cpp enumerates](llamacpp-attention.md), MFA *writes a bespoke kernel*:
head dim, precision, transpose state, even loop bounds become literals. Maximum
specialization, zero shipped binary; the
[runtime-compilation path](../metal/compilation-pipeline.md) makes it free of
Xcode. [Luminal](../mlx/mx-compile.md) is the same school with a search on top.

The [online softmax](../techniques/online-softmax.md) reads like the derivation
with a branch. Note the guard MLX doesn't have:

```swift
    // update 'O'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }
```
— [`AttentionKernel+Softmax.swift:293-298`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Softmax.swift#L293-L298);
row reductions via [`simd_shuffle_xor`](../machine/simdgroup.md), running-sum
update `l = l * correction + l_new` at
[line 321](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Softmax.swift#L321)

Three signature positions distinguish the project:

- **Deliberate [register spilling](../techniques/register-blocking.md).** A third
  blocking dimension along head-dim D, with tile shapes sometimes chosen *to
  spill predictably*: the thesis that a mapped spill beats a smaller tile. The
  boldest register-pressure stance in these case studies.
- **The backward pass exists**, the only open-source FA backward on the
  platform, split into dQ and dK/dV kernels because
  [float atomics are emulated](../machine/special-paths.md); each kernel owns its
  outputs outright.
- **Tile staging on the [dead intrinsics](gemm-async-ghost.md)**:
  `createMetalSimdgroupEvent()` re-declares the async-copy symbols
  ([`GEMMHeaders.swift:24+`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/GEMM/GEMMHeaders.swift#L24)),
  with a long comment documenting an M1-era hardware bug where an async copy whose
  result is never read hangs the GPU until reboot. Pre-Metal-4 code; the repo is
  unmaintained since 2024, but its production descendant (MFA 2.0, C++, inside
  [Draw Things](https://engineering.drawthings.ai/)'s ccv) ships int8 and
  sparse-indexed attention to a real user base.

Companion reading: llama.cpp
[PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021), where Gerganov
builds his FA kernel across 154 public comments. The best line-by-line narration
of these decisions being made anywhere.

Next: [llama.cpp attention](llamacpp-attention.md)
