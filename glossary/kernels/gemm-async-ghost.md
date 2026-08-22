# Case Study: The Async-Copy Ghost

**`async_copy.metal` from
[0xekez/metal-matmul](https://github.com/0xekez/metal-matmul) (BSD-3-Clause) —
the original of the [GEMM lineage](gemm-tiled.md), built on
[`simdgroup_async_copy`](../metal/simdgroup-async-copy.md), kept in the glossary
as the worked example of what DMA-staged tiling looked like and why it died.**

The kernel is the [tiled GEMM](gemm-tiled.md) with the
[cooperative load](../techniques/cooperative-load.md) replaced by hardware DMA.
The declaration half — [hand-declaring compiler intrinsics via `__asm`
linkage](../metal/simdgroup-async-copy.md) — is quoted on the concept page. The
use site is the interesting half:

```metal
    if (s_pos==0) {
      thread _simdgroup_event_t* events[2];
      events[0] = simdgroup_async_copy<TILE_K*8,SW*SIMD_TILE*8>(
        A, a_pos, ushort2(k,n), A_tg);
      events[1] = simdgroup_async_copy<SW*SIMD_TILE*8,TILE_K*8>(
        B, b_pos, ushort2(m,k), B_tg);
      __metal_wait_simdgroup_events(2,events);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
```
— [`async_copy.metal:101-115`](https://github.com/0xekez/metal-matmul/blob/04e80810bbf7ba96ebe26ff84a346d179ee50888/async_copy.metal#L101-L115), abridged

Read `if (s_pos==0)`: **one simdgroup issues the copy for the whole threadgroup**,
fires both transfers, waits on the events; everyone else just waits at the
barrier. The essay's counterintuitive measurement: this was the *fastest*
arrangement — the DMA engine does the moving, so parallelizing the request across
simdgroups bought nothing. Compare the CUDA evolution toward TMA (one thread
issues a bulk tensor copy): same conclusion, officially supported.

**The death.** The Metal-4 frontend (macOS 26) rejects every `__asm("air.*")`
declaration — `error: illegal string literal in 'asm'` — and IR-level workarounds
crash the backend compiler. The m5-gemm port's README documents the failed
resurrection attempts; nothing public replaces the intrinsic. Hence the two
successor kernels: [cooperative loads](gemm-tiled.md) for the staging,
[double buffering](gemm-double-buffered.md) to win back the overlap through ILP.

Why keep reading dead code:

- **The concept outlives the API.** A copy engine with event-signaled completion
  is how [`cp.async`/TMA](../metal/simdgroup-async-copy.md) works, how
  [MFA's generated headers](mfa-codegen.md) staged tiles (with a documented
  M1-era hardware bug: an unread async copy hangs the GPU until reboot), and
  plausibly how a future public Metal API will look. You'll recognize it in one
  glance having read this.
- **It's a lesson in platform risk.** Every project in this glossary that built
  on the undocumented intrinsic — the essay, MFA, early MLX steel — carried
  breakage risk that Metal 4 cashed in. The
  [war stories](../war-stories/cheap-tricks.md) run the same trade knowingly
  (undocumented sysctls, private APIs); this kernel is what the downside looks
  like.

Next: [Steel's BlockLoader](steel-blockloader.md)
