# Cheap Tricks

**Three measured, zero-code or near-zero-code wins that belong in every Apple
Silicon practitioner's kit, plus the reminder that they carry the platform's
signature risk.**

**Raise the GPU wired-memory ceiling.** By default macOS caps how much of
[unified memory](../machine/unified-memory.md) the GPU may wire;
`sudo sysctl iogpu.wired_limit_mb=N` raises it
([the classic thread](https://www.reddit.com/r/LocalLLaMA/comments/186phti/)).
A 192 GB machine goes from ~140 GB to ~184 GB of usable "VRAM", the difference
between a 70B model fitting or not. Undocumented, resets on reboot, occasionally
moved between sysctl names across macOS versions: the
[async-copy story](../kernels/gemm-async-ghost.md) in miniature, run knowingly.

**Read the bandwidth line on the spec sheet before profiling anything.** Across
M-series tiers, decode tokens/sec tracks memory bandwidth almost linearly
([M5 Max megathread](https://www.reddit.com/r/LocalLLaMA/comments/1rqnpvj/)),
because [decode is bandwidth](../techniques/decode-vs-prefill.md). Practical
uses: hardware buying decisions (the Max's 4× bandwidth over the base chip buys
~4× decode; extra GPU cores beyond that buy
[prefill](../techniques/decode-vs-prefill.md), not chat latency), and sanity
checks (if your tok/s is far off `bandwidth ÷ model bytes`, something is broken:
[go look](../metal/profiling.md)).

**Control thermals or your benchmark lies.** MTPLX measured long-run throughput
decay dropping from **50% to 6.7%** just by locking fans to max. Laptop and
fanless chips throttle under sustained ML load; a benchmark's first ten seconds
and its steady state are different machines. Fan control (or a Mac Studio) is
part of [measurement methodology](../techniques/roofline.md) here, not an
enthusiast quirk.

And one honorable mention that rounds out the platform picture: the **ANE
exists**. The Neural Engine, the third matrix engine on the die after the GPU
and [AMX](../machine/amx.md), reaches ~6.6 TFLOPS/watt (vs ~1 for the GPU) but
speaks only Apple's model frameworks (Core ML, superseded for generative
workloads by Core AI in the macOS 27 cycle, which adds a documented hook for
custom Metal kernels); the community's
[reverse-engineering effort](https://www.reddit.com/r/LocalLLaMA/comments/1rhx5pc/)
to train on it through private APIs is equal parts impressive and cautionary.
For Metal work its main relevance is knowing that some workloads
(quantized-conv-shaped inference at low power) have a better home than the GPU,
and that Apple's own frameworks may route there without telling you.

That's the glossary. [SOURCES.md](../../SOURCES.md) has the full annotated list
of repos, writeups, and people worth following to stay current.
