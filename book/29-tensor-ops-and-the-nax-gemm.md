# Tensor ops and the NAX GEMM

## Where we are

Chapter 28 introduced the M5 neural accelerators, a real matrix unit in every GPU
core, reached through Metal 4 tensor operations. This chapter is the programming
model and a kernel written against it: the fourth school of matrix multiply in
this book, after the hand-placed kernel of Part 4, the generated kernels of
metal-flash-attention and the enumerated kernels of llama.cpp. Where the other
three hand-place every fragment, this one describes the multiply and lets the
library and the hardware place it. The book ends where it began, with a matrix
multiply, and with the parts of the old design that the new hardware keeps.

## The programming model: tensors, cooperative tensors, and MPP

Metal 4 adds tensors as a first-class type, `MTLTensor` on the host and
`tensor_inline` and cooperative tensors in shaders, plus Metal Performance
Primitives, MPP, a shader-side library of matrix operations. Together they are
the programming model for the neural accelerators. In CUDA terms it combines
three things in one API: TMA (hardware-managed tile movement), `wmma`
descriptors, and CUTLASS as a vendor library.

- **`MTLTensor`**: a host-side multidimensional resource with rank, extents and
  strides, replacing hand-rolled buffer plus stride arithmetic. Auxiliary planes
  carry block-scaling metadata, the scales per block, for the fp4 and fp8
  formats of Chapter 23.
- **Shader-side tensors**: `tensor_inline` views over device memory or threadgroup
  memory, indexed and sliced in the shading language.
- **Cooperative tensors**: a tensor whose storage is distributed across an
  execution group's registers, the way `simdgroup_matrix` fragments are, Chapter
  12, but sized and shaped by a descriptor rather than fixed at 8 by 8.
- **MPP**: `mpp::tensor_ops::matmul2d` and its relatives. You declare a
  `matmul2d_descriptor` with the shapes, data types, transpose and accumulate
  mode; pick an execution scope, `execution_thread`, `execution_simdgroup`,
  `execution_simdgroups<N>` or `execution_threadgroup`; call the operation; and
  receive the destination as a cooperative tensor from
  `get_destination_cooperative_tensor()`. The primary references are session
  262 of WWDC 2025, session 330 of WWDC 2026, and chapter 7 of the Metal Shading
  Language 4.1 specification.

This model belongs next to the copy engine of Chapter 14, because it continues
that story. Cooperative tensors let an operation read operands from device memory
and land results in distributed registers without the manual threadgroup-memory
staging of Chapter 11. On this hardware the descriptor-and-scope model does the
job the copy intrinsic once did, plus the job the tile loop did. How far that
erodes the classic tiling shape on M5 is not yet publicly established. Parts 4
and 5 describe the shape every shipped pre-NAX kernel uses, and this chapter
describes the fork to watch.

If you port code, note that this API is macOS 26 and M5-era surface. Everything
else in Part 2 runs back to M1, so tensor-operation code needs a fallback path.
MLX structures its parallel NAX steel that way.

## Two implementations, read together

**MLX's NAX steel**, `steel/gemm/nax.h`, MIT, rebuilds the register-blocked
multiply of Chapter 18 on 16 by 16 fragments; Chapter 28 quoted its fragment
type. The file opens with the MPP include and wraps the library's tensor
operations in the same loader, multiply and epilogue decomposition that steel
uses everywhere else. The decomposition survived the hardware change in outline: the kernel still
wires a loader, a multiply and an epilogue. The parts changed more than the
outline: the multiply layer moved to the tensor operations with 16 by 16
fragments, the loader no longer stages operands through threadgroup memory,
and the epilogue is reimplemented for the new tile. You can see this by diffing
a `_nax` kernel against its plain sibling.

**flash-moe's standalone kernel**, `metal_infer/nax_gemm.metal`, about 230
lines, shows the API without the framework. It declares an
`mpp::tensor_ops::matmul2d_descriptor` with the problem shape and chooses an
`execution_simdgroups<4>` scope. That scope is four cooperating simdgroups, the
same threadgroup of four simdgroups that Part 4 used, now working on one
hardware-placed multiply. The kernel then obtains the accumulator with
`get_destination_cooperative_tensor` and iterates the shared dimension. Its
comments document the part that causes porters trouble: the tensor operations
assume column-major layouts, and getting row-major language-model weights
through them is layout work that the descriptor does not hide.

## The fourth school against the other three

- **Against hand-placed steel**, Chapter 18: the kernel has no `simdgroup_load`
  choreography and no per-fragment stride constants. The descriptor states the
  shape and the library handles placement. You give up the register-blocking
  dials this book spent Chapters 2, 4 and 12 on, and you get the new hardware's
  throughput in exchange.
- **Against generated kernels**, metal-flash-attention in Chapter 21:
  specialization moves from generated source into descriptor parameters, and the
  shader is generic and small.
- **Against enumeration**, llama.cpp in Chapter 21: llama.cpp's tensor path
  coexists with its classic kernels behind run-time checks, the same fallback
  discipline MLX uses, where NAX steel compiles alongside plain steel and the
  dispatch of Chapter 17 picks per hardware.

This school is the youngest, and its tuning is visibly in flux: recent MLX
commits adjust NAX tile picks and batch limits release by release. Nothing here
runs on pre-M5 hardware. Treat it as the direction of travel. The tiled matrix
multiply of Part 4 is still the foundation course.

## Where the matmul stands, and where the book ends

The example began as a library call at 11.7 TFLOPS. It became a 144-line kernel
at 13.5, at the practical ceiling of the regular pipes on an M5 Max; the same
decomposition in production inside steel; the shape that attention takes when
its intermediate is a whole matrix; a decode step whose time is a byte count; and
now a descriptor handed to a matrix unit that did not exist when the kernel was
written. Some things stayed constant across all of these: the output lived in registers,
the inputs passed through a thin staging buffer, the win came from moving fewer
bytes, and the measurement decided.

## Check yourself

1. What does a `matmul2d_descriptor` replace from the hand-written kernel of
   Part 4, and what does the kernel writer give up?
2. Why does MLX keep the plain steel kernels alongside the NAX fork rather than
   replacing them?
3. In the steel decomposition, what changed for NAX and what held, and why does
   that matter?

### Answers

1. The tile constants, the fragment loop and the per-fragment stride arithmetic:
   the descriptor states the shape and the library places the fragments. The
   writer gives up the register-blocking dials, the tile size and accumulator
   grid choices that Chapters 2, 4 and 12 tuned.
2. Because tensor operations run only on M5-era hardware and macOS 26 or later,
   while everything else runs back to M1, so the dispatch picks the NAX path on
   hardware that has it and falls back otherwise.
3. The multiply layer moved to hardware-placed 16 by 16 fragments, the loader
   stopped staging through threadgroup memory, and the epilogue was rewritten
   for the new tile; what held is the decomposition itself, loader, multiply,
   epilogue and a wiring kernel. It matters because it shows which part of the
   design was the hardware's and which was the architecture's.

That is the book. The glossary holds the same material as standalone entries,
and the sources file lists the repositories, write-ups and people to follow to
stay current.
