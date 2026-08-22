# simdgroup_matrix

**`simdgroup_matrix<T, 8, 8>` is Metal's matrix-multiply primitive: an 8×8 matrix
held collectively in one [simdgroup's](../machine/simdgroup.md) registers, with a
single-call multiply-accumulate.**

CUDA equivalent: warp-level MMA — `wmma` fragments / `mma.sync` — as an API.
As hardware, the analogy is looser: pre-M5 Apple GPUs have no separate tensor-core
unit; `simdgroup_matrix` ops execute on the regular FP32 pipes, arranged well
(the reverse-engineered [Rigel paper](https://arxiv.org/abs/2606.12765) confirms
even Metal 4.1's newer `matmul2d` tensor API has no dedicated matrix datapath on
M4-class hardware). The performance story is therefore different from NVIDIA's:
you use `simdgroup_matrix` for its *register layout and issue efficiency*, not for
a 10× throughput unlock — and a well-written simdgroup-matrix GEMM
[still reaches 13.5 TFLOPS fp32 on an M5 Max, beating MPS](../kernels/gemm-tiled.md).

The API in one breath — load fragments, accumulate, store:

```metal
simdgroup_float8x8 A_simd, B_simd;
#pragma clang loop unroll(full)
for (ushort i = 0; i < DIM * 8; i += 8) {
  simdgroup_load(A_simd, A, DIM * 8, ulong2(i, c_pos.y));
  simdgroup_load(B_simd, B, BN,      ulong2(c_pos.x, i));
  simdgroup_multiply_accumulate(acc, A_simd, B_simd, acc);
}
```
— [m5-gemm `sync_copy.metal:29-35`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L29-L35)

Rules of use, all matching your `wmma` instincts:

- **Fragments are opaque.** You never index into a `simdgroup_float8x8`; the 64
  elements live distributed across the 32 threads in an undocumented layout.
  `simdgroup_load`/`simdgroup_store` move whole tiles between the fragment and
  [threadgroup](../machine/threadgroup-memory.md) or device memory.
- **They are [registers](../machine/registers.md), and they add up fast.** A 4×4
  grid of accumulators is 1024 floats per simdgroup; doubling that
  [spilled and ran 10× slower](../machine/registers.md) in the measured case. The
  accumulator budget *is* the tile-size decision.
- **Types:** `simdgroup_float8x8`, `simdgroup_half8x8`, `simdgroup_bfloat8x8` —
  [F16 fragments halve the register cost](../machine/f16.md); serious kernels take
  16-bit inputs and [accumulate in fp32](../machine/f16.md).

Everything above `simdgroup_matrix` is composition: MLX's
[`BlockMMA`](../kernels/steel-blockmma.md) tiles these 8×8 fragments into
simdgroup-level register blocks; [attention kernels](../kernels/steel-attention.md)
chain two of those through an [online softmax](../techniques/online-softmax.md).
When Q has only one row and no 8×8 tile can be filled, kernels abandon
`simdgroup_matrix` entirely — the [decode-vs-prefill split](../techniques/decode-vs-prefill.md).

Next: [simdgroup_async_copy](simdgroup-async-copy.md)
