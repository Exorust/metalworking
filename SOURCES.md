# The annotated source list

Everything the track draws from, plus the feeds for staying current. All URLs verified
2026-08-07.

## Start here (top 5 across everything)

1. **[philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)** - the reverse-engineered M-series GPU spec sheet Apple never published. ~208 KB register file/core, FFMA32 dual-dispatches 2 inst/cycle, ALU saturates at 24 SIMDs/core, F16 wins because register-dependency stalls are 1.56 vs 1.84 cycles.
2. **[Fast Matrix Multiply on Apple GPU](https://percisely.xyz/gemm)** + repo [0xekez/metal-matmul](https://github.com/0xekez/metal-matmul) - the canonical naive-to-fast Metal GEMM tutorial. ~2.5 TFLOPS fp32 on a MacBook Air, beats MPS, via undocumented `simdgroup_async_copy` (counterintuitively fastest when ONE simdgroup does all the loading). Metal 4 / M5-era port: [yaroslavvb/m5-gemm](https://github.com/yaroslavvb/m5-gemm) (13.5 TF vs MPS 11.7; the async copy intrinsic no longer compiles, replaced with cooperative loads + double buffering).
3. **[llama.cpp Flash Attention PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021)** - Gerganov walks through a Metal FA kernel line by line in 154 comments (threadgroup layout, simdgroup_load strides, online softmax). Companion: [PR #2615](https://github.com/ggml-org/llama.cpp/pull/2615) which deleted MPS for hand-written simdgroup_matrix quantized kernels (~88% ALU on 4096^2 matmul; honest caveat: end-to-end only ~40% because non-matmul ops dominate).
4. **[Engineering @ Draw Things](https://engineering.drawthings.ai/)** (Substack) - the best ongoing Metal-perf publication. Highlights: [making the ANE work](https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in) (~22 of the advertised 38 TFLOPs unlocked via native int8 + IOSurface), hand-written implicit-GEMM 3D conv beating MPSGraph 4-8.3x, and MFA 2.0 (94% faster than ggml).
5. **r/LocalLLaMA + [@awnihannun](https://x.com/awnihannun) + [@zcbenz](https://x.com/zcbenz)** - the live feed. r/LocalLLaMA is effectively the only high-signal subreddit; awnihannun posts MLX perf milestones; zcbenz is the deepest on Metal API internals with the best signal-to-follower ratio.

## GitHub

- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) - 83% ALU utilization on M1 Max (4400 GINSTR/s); tricks: third block dim along head-dim D, deliberately controlled register spilling, backward split into dQ and dK/dV kernels because Apple emulates FP32 atomics badly.
- [MLX steel kernels](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels/steel) + [custom kernel API](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) - CUTLASS-style templated Metal GEMM/attention library; `mx.fast.metal_kernel` fused grid_sample 8x fwd / 40x bwd. [PR #3018](https://github.com/ml-explore/mlx/pull/3018): Split-K GEMM on M5 neural accelerators, up to 1.62x.
- [luminal-ai/luminal](https://github.com/luminal-ai/luminal) + [Compiling fast GPU kernels](https://docs.luminalai.com/blog/gpu) - the compiler school: static compute graph, primitive ops swapped for generated MSL then pattern-matched into fused variants, compile-time buffer assignment, all of Llama 3 8B in ONE Metal command buffer, search-based compilation that rediscovers Flash Attention. Q8 Llama 3 8B: 15-25 tok/s on M-series, within ~20% of llama.cpp.
- [dougallj/applegpu](https://github.com/dougallj/applegpu) - Apple G13 ISA disassembler/emulator; patch instruction bytes into compiled shaders and diff GPU vs emulator to see what the compiler actually emitted.
- [tinygrad](https://github.com/tinygrad/tinygrad) + [mesozoic-egg/tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes) - the autotuning school: BEAM search over tiling/unroll/local-size candidates, benchmarked on real hardware.
- [abeleinin/Metal-Puzzles](https://github.com/abeleinin/Metal-Puzzles) - best on-ramp; 14 progressive MSL puzzles via MLX custom kernels.
- [corsix/amx](https://github.com/corsix/amx) - definitive doc of Apple's undocumented AMX matrix coprocessor (32x32 compute grid). Also [philipturner/amx-benchmarks](https://github.com/philipturner/amx-benchmarks).
- Skip: candle/mistral.rs/whisper.cpp kernels (ggml/MLX-derived); llama.cpp PR #13941's "1140%" claim (never merged, numbers don't hold).

## Blogs / papers

- [Rigel: reverse-engineering Metal 4.1 tensor compute on M4 Max](https://arxiv.org/abs/2606.12765) - fp8 is software-emulated (0.94x fp16 throughput despite half the bytes); `matmul2d` has no dedicated matrix datapath.
- [MLX kernel fusion explained](https://nipunbatra.github.io/blog/posts/2026-04-25-mlx-kernel-fusion.html) - 17x on 4kx4k GELU by fusing 8 kernels; why Apple Silicon ML is DRAM-traffic-bound.
- [138 GFLOPS radix-8 Stockham FFT](https://arxiv.org/abs/2603.27569) - beats vDSP by 29%; key inversion of CUDA instinct: threadgroup barriers are cheap (~2 cycles), scattered threadgroup access is the real cost.
- [Writing Fast ML Kernels on Apple Silicon](https://medium.com/@srivarshan02/writing-fast-ml-kernels-on-apple-silicon-123152624078) - 57% -> 86% of memcpy ceiling on M3 Pro via threadgroup caching, float4 loads, hierarchical simd_sum.
- [BAIR: CUDA to MLX / K-Search](https://bair.berkeley.edu/blog/2026/07/29/cuda-to-mlx-k-search/) - which CUDA intuitions transfer (and which are wrong): 32 KB threadgroup mem vs 48, the exp2 hardware trick, ~20x Mamba prefill via parallel scan.
- [Alyssa Rosenzweig's M1 GPU series](https://alyssarosenzweig.ca/blog/asahi-gpu-part-n.html) - the AGX reverse-engineering canon (no native geometry/tessellation hardware at all).
- [BaseRT paper](https://arxiv.org/abs/2607.00501) - 1.56x decode over llama.cpp via chip-specific kernel fusion.
- Methodology check: ["I beat MLX's SDPA... then discovered it was useless"](https://medium.com/@rajveer.rathod1301/i-built-a-flashattention-kernel-that-beat-mlxs-sdpa-then-i-discovered-it-was-useless-e4ce6ebf953c) - 1.2-1.3x synthetic, 0% end-to-end.
- [A Hard Look at Softmax: Torch vs MLX](https://aditvenk.substack.com/p/a-hard-look-at-softmax-torch-vs-mlx) - dissects MLX's hand-tuned softmax shader.

## Twitter/X

Must-follows:

- [@awnihannun](https://x.com/awnihannun) - MLX lead; perf milestones (DeepSeek V3 4-bit >20 tok/s on M3 Ultra 512GB; Kimi K2 1T across two Ultras via mx.distributed).
- [@ivanfioravanti](https://x.com/ivanfioravanti) - most prolific M-series benchmarker; posts prefill/decode deltas per kernel revision.
- [@zcbenz](https://x.com/zcbenz) - MLX maintainer; Metal API internals (Metal4 fp4/fp8 block-scaling, RDMA over Thunderbolt). Underfollowed.

Also: [@ggerganov](https://x.com/ggerganov) (low volume, zero noise), [@__tinygrad__](https://x.com/__tinygrad__) (public Metal debugging, e.g. IndirectCommandBuffer issues on M1/M2), [@alexocheema](https://x.com/alexocheema)/exolabs (Mac cluster scaling), [@Prince_Canuma](https://x.com/Prince_Canuma) (MLX vision/audio), [@DiganiJagrit](https://x.com/DiganiJagrit) (the "kernel magic" behind MLX's 10-200% matmul speedup; rarely posts).

Corrections as of 2026-08: @realGeorgeHotz is wiped (0 tweets); Philip Turner (@philipturnerar) pivoted to nanotech - use his repos; Asahi GPU devs are on Mastodon (@AsahiLinux@treehouse.systems), not X.

## Reddit (r/LocalLLaMA is the only venue that matters)

- [Skipping 90% of KV dequant work: +22.8% decode at 32K](https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/) - 14 micro-optimizations all lost to baseline; the win was deleting work (skip V dequant for near-zero softmax weights, 3 lines).
- [DFlash speculative decoding: 85 tok/s on M5 Max](https://www.reddit.com/r/LocalLLaMA/comments/1simszl/) - every hand-written custom kernel was 0.5-0.8x SLOWER than stock MLX steel; the wins were a head_dim patch to unlock the fast SDPA path and halving GPU->CPU syncs.
- [36 experiments to 20.34 tok/s on a 209GB MoE in 128GB RAM](https://www.reddit.com/r/LocalLLaMA/comments/1s7g8ov/) - temporal expert prefetch (27% expert reuse token-to-token) + command-buffer pre-encode; all 28 failures listed.
- [MTPLX: 2.24x via 10 lines of Metal](https://www.reddit.com/r/LocalLLaMA/comments/1t3zuvy/) - stock MLX quantized matvec stalls at small M; wider simdgroups + unrolling, bit-exact.
- [TurboQuant: 4.6x KV compression](https://www.reddit.com/r/LocalLLaMA/comments/1s5vhf6/) - 0.28x -> 0.98x fp16 speed via fused quant/dequant kernels.
- [ANE reverse-engineered to train a GPT](https://www.reddit.com/r/LocalLLaMA/comments/1rhx5pc/) - ~6.6 TFLOPS/watt at 2.8W (vs ~1 for the GPU); repo [maderix/ANE](https://github.com/maderix/ANE).
- [OpenEvolve: LLM-evolved Metal kernels](https://www.reddit.com/r/LocalLLaMA/comments/1lm98z7/) (+ skeptical [r/ML crosspost](https://www.reddit.com/r/MachineLearning/comments/1lmqbzc/)) - +12.5% avg decode claimed, but huge variance and regressions; the repo's own analysis reports the evolved kernel lost to baseline.
- [M5 Max benchmark megathread](https://www.reddit.com/r/LocalLLaMA/comments/1rqnpvj/) and [M5 vs DGX Spark vs Strix Halo vs RTX 6000](https://www.reddit.com/r/LocalLLaMA/comments/1tfzsd6/) - memory bandwidth predicts tok/s almost linearly.
- Classic hack: [`sudo sysctl iogpu.wired_limit_mb=N`](https://www.reddit.com/r/LocalLLaMA/comments/186phti/) - 192GB machine: ~140GB -> 184GB usable VRAM.
