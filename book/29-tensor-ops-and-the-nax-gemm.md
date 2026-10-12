# Tensor ops and the NAX GEMM

## Where we are

Chapter 28 showed you the M5 neural accelerators, a real matrix unit in every GPU
core, reached through Metal 4 tensor operations. What you have not yet seen is the
programming model, or a kernel written against it. That kernel is the fourth
school of matrix multiply in this book, after the hand-placed kernel of Part 4,
the generated kernels of metal-flash-attention and the enumerated kernels of
llama.cpp. The other three hand-place every fragment. This one describes the
multiply and lets the library and the hardware place it. The book ends where it
began, with a matrix multiply, and with the parts of the old design that the new
hardware keeps.

## The programming model: tensors, cooperative tensors, and MPP

What do you write to reach a matrix unit? Metal 4 adds tensors as a first-class
type, `MTLTensor` on the host and `tensor_inline` and cooperative tensors in
shaders, plus Metal Performance Primitives, MPP, a shader-side library of matrix
operations. Together they are the programming model for the neural accelerators.
In CUDA terms it is a loose blend of three separate things: TMA (hardware-managed
tile movement), `wmma` fragments (matrix operands that live in registers), and
CUTLASS as a vendor library.

On the host, an `MTLTensor` is a multidimensional resource with rank, extents and
strides, and it replaces the buffer plus stride arithmetic you would otherwise
hand-roll. Auxiliary planes carry block-scaling metadata, the scales per block,
for the fp4 and fp8 formats of Chapter 23. In the shader, `tensor_inline` gives
you views over device memory or threadgroup memory, indexed and sliced in the
shading language.

A cooperative tensor goes further. Its storage is distributed across an execution
group's registers, the way `simdgroup_matrix` fragments are, Chapter 12, but a
descriptor sizes and shapes it rather than fixing it at 8 by 8.

MPP is where the multiply lives: `mpp::tensor_ops::matmul2d` and its relatives.
You declare a `matmul2d_descriptor` with the shape of one operation, its M, N and
K, the transpose flags and the accumulate mode; the element types come from the
tensor types and template arguments, not from the descriptor. You pick an
execution scope, `execution_thread`, `execution_simdgroup`,
`execution_simdgroups<N>` or `execution_threadgroup`. You create the destination
as a cooperative tensor with `get_destination_cooperative_tensor()`, and then you
call `run`, which writes into it. The primary references are session 262 of WWDC
2025, session 330 of WWDC 2026, and chapter 7 of the Metal Shading Language 4.1
specification.

What happens to the staging you built by hand? This
model belongs next to the copy engine of Chapter 14, because it continues that
story. Cooperative tensors can take operands from device memory or from
threadgroup memory and keep results in distributed registers. How much of the
staging of Chapter 11 that removes depends on the kernel, and the two
implementations below disagree: MLX's NAX loader reads device memory straight
into fragments and keeps a K loop and fragment loops, while flash-moe's kernel
still copies tiles into threadgroup arrays behind barriers. The
descriptor-and-scope model replaces the multiply. By itself it does not replace
the loader. Parts 4 and 5 describe the shape every shipped pre-NAX tiled matrix
multiply uses; what you are reading here is the fork to watch.

Before you port code, separate what the API needs from what pays. It needs macOS
26 and a Metal 4 GPU. The pinned llama.cpp compiles its tensor path on M2 Ultra
and M4 and turns it off there by default, because it measured no gain: about 5
percent slower on M2 Ultra and no significant difference on M4. The hardware
that makes the API worth using is M5. Everything else in Part 2 runs back to M1,
so your tensor-operation code needs a fallback path. MLX structures its parallel
NAX steel that way, and its dispatcher adds conditions of its own: macOS 26.2, a
NAX-enabled build, non-complex operands, and TF32 on for 32-bit inputs, Chapter
17.

## Two implementations, read together

A specification tells you what the calls are. Two kernels show you what they cost
in practice: one inside a framework, one standing alone.

MLX's NAX steel, `steel/gemm/nax.h`, MIT, rebuilds the register-blocked multiply
of Chapter 18 on 16 by 16 fragments; Chapter 28 quoted its fragment type. The
file opens with the MPP include and wraps the library's tensor operations in the
same loader, multiply and epilogue decomposition that steel uses everywhere else.
The decomposition survived the hardware change in outline: the kernel still wires
a loader, a multiply and an epilogue.
The multiply layer moved to the tensor operations with 16 by 16 fragments, the
loader no longer stages operands through threadgroup memory, and the epilogue is
reimplemented for the new tile. You can see this by diffing a `_nax` kernel
against its plain sibling.

The tile is still yours. `NAXTile` holds a grid of fragments sized by template
parameters, `tile_matmad_nax` loops over that grid, and the fused kernel is
instantiated for six tile shapes, four transpose combinations and three element
types, the same enumeration as plain steel. Inside a fragment the lane mapping is
explicit too: `get_coord()` computes which rows and columns each lane owns. What
the library takes over is the multiply itself.

flash-moe's standalone kernel, `metal_infer/nax_gemm.metal`, about 230 lines,
shows you the API without the framework. It declares an
`mpp::tensor_ops::matmul2d_descriptor` with its tile shape, 32 by 32 by 32, not
the problem's M, N and K, and chooses an `execution_simdgroups<4>` scope. That
scope is four cooperating simdgroups, the same threadgroup of four simdgroups that
Part 4 used, now working on one library-executed multiply. The kernel creates the
accumulator with `get_destination_cooperative_tensor`, then for each 32-deep slab
of K copies the A and B tiles into threadgroup arrays, zero-filling past the
matrix edges, waits at a barrier, and calls `run` on tensors that view those
arrays.

Its comments document the part that causes porters trouble. The tensor operations
assume column-major layouts, so getting row-major language-model weights through
them is layout work the descriptor does not hide, and the output comes back
column-major too; the file ships a helper, `nax_extract_row0`, just to read the
first row out.

## The fourth school against the other three

Where does this leave the three schools you already know? Against the hand-placed
steel of Chapter 18, the kernel has no `simdgroup_load` choreography. The
fragment's lane layout is still spelled out in `get_coord()`, and the tile size
and accumulator grid are still yours, so the register-blocking choices of
Chapters 2, 4 and 12 remain. What you give up is the multiply-accumulate itself
and the option of running it on the regular pipes. What you get is the matrix
unit.

Against the generated kernels of metal-flash-attention in Chapter 21, some
specialization moves into descriptor parameters, but MLX still instantiates the
NAX kernel per shape, transpose and element type, as plain steel does.
flash-moe's shader, with one fixed tile, is the generic and small one.

Against the enumeration of llama.cpp, also in Chapter 21, the fallback discipline
is shared. llama.cpp's tensor path coexists with its classic kernels behind
run-time checks, the same discipline MLX uses, where NAX steel compiles alongside
plain steel and the dispatch of Chapter 17 picks per hardware.

This school is the youngest, and its tuning is visibly in flux: recent MLX commits
adjust NAX tile picks and batch limits release by release. The hardware it targets
is M5; on earlier chips the API may compile and does not pay. Treat it as the
direction of travel. The tiled matrix multiply of Part 4 is still the foundation
course.

## Where the matmul stands

What carried through every version of the matrix multiply you followed, and what
did not? The example began as a library call at 11.7 TFLOPS. It became a 144-line
kernel at 13.5, at the practical ceiling of the regular pipes on an M5 Max; the
same decomposition in production inside steel; the shape that attention takes
when its intermediate is a whole matrix; a decode step whose time is a byte count
divided by bandwidth; and now a descriptor handed to a matrix unit that the
kernel's design predates. Two things held across all of these: the output lived
in registers, and the measurement decided.

Two did not. The thin staging buffer disappears in MLX's NAX loader and stays in
flash-moe's. And the wins of Parts 4 to 6 came from moving fewer bytes, while the
matrix unit's win is more arithmetic per second, the exception Chapter 3 made for
large matrix multiply.

## Review the content

1. What does a `matmul2d_descriptor` replace from the hand-written kernel of
   Part 4, and what do you give up when you use it?
2. Why does MLX keep the plain steel kernels alongside the NAX fork rather than
   replacing them?
3. In the steel decomposition, what changed for NAX and what held, and why does
   that matter?

### Answers

1. The `simdgroup_load` and `simdgroup_multiply_accumulate` calls: the
   descriptor states one operation's shape and the library executes it. You keep
   the tile size, the accumulator grid and the fragment loops, and you give up
   the multiply itself and the option of running it on the regular pipes.
2. Because the matrix unit exists only on M5-class hardware and the path needs
   macOS 26.2, a NAX-enabled build and, for 32-bit inputs, TF32, while plain
   steel runs back to M1; the dispatch of Chapter 17 checks all of that and
   takes the plain path whenever any condition fails, including on an M5 with
   TF32 off.
3. The multiply layer moved to library-executed 16 by 16 fragments whose lane
   layout the code still spells out, the loader stopped staging through
   threadgroup memory, and the epilogue was rewritten for the new tile; what
   held is the decomposition itself, loader, multiply, epilogue and a wiring
   kernel, and the tile choices inside it. It matters because it separates what
   the hardware changed from what the architecture kept.

That is the book; the glossary holds the same material as standalone entries,
and the sources file lists the repositories, write-ups and people to follow to
stay current.
