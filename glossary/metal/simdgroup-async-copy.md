# simdgroup_async_copy

**`simdgroup_async_copy` was Apple's undocumented DMA intrinsic for asynchronous
tile copies from device to threadgroup memory, and it no longer compiles. It's in
this glossary because its ghost explains the shape of current kernels.**

CUDA equivalent: `cp.async` / TMA, a copy engine that moves tiles while the ALUs
compute, completion signaled by events. Apple's hardware has such an engine, but
Apple never exposed it in MSL. The compiler exposed it anyway, and the community
found it: declare the intrinsic by its AIR symbol name via inline-asm linkage,

```metal
thread _simdgroup_event_t* __metal_simdgroup_async_copy_2d(
  ulong,               // sizeof(element)
  ulong,               // alignof(element)
  threadgroup void *,  // dst
  ...
  __asm("air.simdgroup_async_copy_2d.p3i8.p1i8");

void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");
```
— [metal-matmul `async_copy.metal:9-28`](https://github.com/0xekez/metal-matmul/blob/04e80810bbf7ba96ebe26ff84a346d179ee50888/async_copy.metal#L9-L28), abridged

then use it like `cp.async`: kick off the copy, compute, wait on the event. The
counterintuitive finding from the [original GEMM essay](https://percisely.xyz/gemm):
it was fastest when a *single simdgroup* issued the copy for the whole threadgroup,
because the DMA hardware does the moving and parallelizing the ask bought nothing.
[metal-flash-attention](../kernels/mfa-codegen.md) built its tile staging on the
same intrinsics, and documents an M1-era hardware bug where an async copy whose
result is never read **hangs the GPU until reboot**. Undocumented-API life.

**Metal 4 (macOS 26) closed the door**: the compiler rejects every `__asm("air.*")`
declaration outright, and LLVM-IR-level workarounds crash the backend. There is no
public replacement; Metal 4's `<metal_cooperative_tensor>` ships only a generic
layout interface, not a DMA primitive. Full post-mortem in the
[async-ghost case study](../kernels/gemm-async-ghost.md).

What fills the gap on current toolchains:
[cooperative loads](../techniques/cooperative-load.md)
(plain per-thread vectorized copies) plus
[double buffering](../techniques/double-buffering.md), which recovers most of the
overlap through instruction-level parallelism instead of a copy engine. The same
threads issue loads and math, but with no data dependence between them, the
scheduler interleaves. That substitution, DMA overlap traded for ILP overlap, is
the single clearest example of why current Apple kernels look different from their
CUDA cousins, and why a future public async-copy API would reshape them again.

Next: [MPS](mps.md)
