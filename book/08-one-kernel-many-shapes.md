# One kernel, many shapes

## Where we are

You compiled one kernel into one pipeline state in Chapter 6, and you launched it
in Chapter 7. In production, what looks like one kernel is several. One general matrix multiply can handle both
the case where the matrix divides evenly by the tile and the ragged edge, but it
pays for the checks on every tile; a specialized variant per case does not. The
same holds for an attention kernel with and without a mask, and for a quantized
kernel per weight format. In CUDA you produce the variants ahead of time or at run
time. Metal gives you a third option, and it follows from Chapter 6: the backend
compile normally runs on the user's device, with the host's knowledge available.

## The problem, and CUDA's two answers

How do you get a kernel that contains only the code one configuration needs, so
that no branch on the configuration survives into the hot loop? That is
specialization. The two common CUDA routes are template instantiation ahead of time, where
the fatbin holds one instance per configuration, and NVRTC, which compiles source
at run time for the configuration at hand. In both, each variant goes through the
compiler as its own kernel.

Metal does not force that choice on you.

Its backend compile normally runs on the device, inside the call that makes
the pipeline state. That allows a middle route: ship one function in AIR, the
portable intermediate representation, with declared constant slots; bind values to
the slots when you build the pipeline state; and let the backend compiler fold
them, resolving the branches that depend on them and deleting the code on the dead
side. Each new combination of values is still a backend compile, unless a cached
pipeline for it exists; what is saved is the source compile and the shipping of
one binary per variant. These slots are function constants. On the host, the
sequence is: fill a `MTLFunctionConstantValues` object, ask the library for the
function specialized with those values, and make the pipeline state from that
specialized function.

## What they look like

What does a function constant look like in a kernel you might read? MLX's fused
matrix-multiply kernel, which Part 5 reads in full, declares these function
constants.

```cpp
constant bool has_batch [[function_constant(10)]];

constant bool use_out_source [[function_constant(100)]];
constant bool do_axpby [[function_constant(110)]];

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];
```
Source: [`steel_gemm_fused.h:9-16`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h#L9-L16)

Each is a `constant bool` with a numbered slot. The kernel body branches on them
like ordinary variables. When the host knows that the output divides evenly by
the tile in both dimensions, it binds `align_M` and `align_N` to true, and the
compiled pipeline's main loop contains only the unchecked, vectorized loads, with
no edge handling for those two dimensions.

`align_K` is a separate constant, because the shared dimension can be ragged on
its own, and when it is false the kernel keeps a bounds-checked tail step.

When `align_M` and `align_N` are false, the pipeline keeps both paths: checked
loads for the edge tiles, and unchecked loads, selected at run time, for the full
tiles in the interior. So the aligned pipeline carries none of the edge code, and the ragged pipeline
carries both. Chapter 18 shows the two paths side by side.

The attention kernels use the same mechanism for causality and masking. `do_causal`
and `has_mask` are independent function constants in MLX's attention kernel, so a
pipeline built for causal attention without an explicit mask carries the causal
bound and none of the mask loads. llama.cpp's `FC_flash_attn_ext_*` family is the
same pattern at production scale: its one shader file declares function constants
for whether a mask is present and for the block sizes of its mask-precompute
kernel, among others, and the host builds a pipeline per combination it meets.

A branch on a configuration flag is uniform, the same for every lane, so it costs
instruction overhead rather than lane divergence on either platform. The constants
remove that overhead and the dead side's code. A branch on data still diverges in
the usual way (Chapter 1).

## The three channels, and which to use

When you have a configuration value in hand, where should it go? It reaches a
Metal kernel through one of three channels, fixed at three different times, and
the right one depends on what the value affects.

| Channel | Fixed when | Use it for |
|---|---|---|
| C++ templates and `-D` defines | before AIR, when MSL is compiled | anything the source itself depends on: tile shapes that size `threadgroup` arrays, data types, template arguments |
| Function constants | when the pipeline state is built, from AIR to binary | flags that select a code path, and occasionally values; llama.cpp passes stride integers this way |
| Buffer arguments | at run time, per dispatch | true runtime values: sizes, pointers |

A tile shape sizes a `threadgroup` array and appears in template types, so the
source cannot be compiled without it; it has to be a template parameter or a
define. Steel specializes its tile shapes with templates, and the matrix multiply
this book follows sets its tile sizes with `-D` defines (Chapter 11). A flag that
only selects between code paths does not change the source's types, so it can wait
for the backend. The backend allocates registers for the specialized code it
produces (Chapter 1), and a constant that removes a path can change that
allocation. A value that changes per dispatch goes in a buffer and costs nothing
at compile time.

One shipped AIR function therefore yields many pipelines, and you cache them,
keyed by the function and the combination of constant values, because building a
pipeline state is the expensive step of Chapter 6. Every new combination is a
backend compile on the user's machine, so the number of combinations a kernel can
meet is a cost to watch. Chapter 17 shows how MLX keys and caches its pipelines.

## The attribute that belongs with them

You have already met a second promise to the compiler. The kernel attribute
`max_total_threads_per_threadgroup`, which Chapters 1, 2 and
7 have each pointed at, is a separate mechanism that works on the same idea. It
tells the compiler an upper bound the run time promises to respect, so that the
register allocator can plan for that many threads instead of the hardware maximum.
Function constants tell the compiler which code path the run time will take, so
that it can delete the others. Both are contracts between the host and the
backend compile, and both need their information before code generation; CUDA
has the same bound in `__launch_bounds__`.

That is the least Metal you need: a pipeline state from source, a dispatch in a
command buffer, a wait, and the contracts that let one kernel serve many shapes.
Part 3 uses exactly this much to measure a kernel before anyone optimizes it.

## Check yourself

1. A CUDA kernel is a template instantiated for `ALIGNED = true` and `ALIGNED =
   false`, and both instances ship in the fatbin. What is the Metal equivalent,
   and what ships?
2. A kernel's tile height sizes its `threadgroup` staging array and is a template
   argument of its loader. The host knows it before building the pipeline. Which
   channel carries it, and why not a function constant?
3. A framework meets a new combination of function constants at run time. What
   does that cost, and what does the framework do about it?

### Answers

1. One AIR function with an `align` function constant. The metallib ships that
   single function; the two pipelines are built on the device, the aligned one
   with only the unchecked path and the ragged one with the checks and the
   unchecked path for interior tiles.
2. A template parameter or a `-D` define. The source needs the value to
   declare the array and instantiate the loader. A function constant's value is
   bound only after the source has been compiled to AIR.
3. A backend compile when the pipeline state is built, unless a cached pipeline
   exists. The framework caches each pipeline keyed by the function and the
   constant values, so the cost is paid once per combination for as long as the
   cache lives.

The next part measures: what to beat, how to time a kernel, and how to turn a time
into a verdict.

Next: [The baseline and the stopwatch](09-the-baseline-and-the-stopwatch.md)
