# When it is still slow: reading the machine code

## Where we are

The methodology of Chapter 9 had four steps. Timestamps and arithmetic,
differential benchmarking, and the hygiene rules carry most kernels to an
answer. This chapter is the remaining step, the third in Chapter 9's list,
disassembly when the arithmetic says impossible. When the roofline of Chapter 10 says a
kernel should be fast and it is not, and changing one thing at a time has not
found the lever, the last resort is reading what the compiler emitted. Apple
ships no tool for that. You do it with the reverse-engineered applegpu
toolchain, which is also how the community learned most of what Part 1 reported.

## What you have, compared with CUDA

On NVIDIA, `cuobjdump` and `nvdisasm` read SASS, the vendor documents the tool,
and about half of the instruction set is documented. Here the instruction set of
the Apple G13 family, the M1 generation, with coverage of later generations
growing, was reverse engineered by the community. The on-device final compile
of Chapter 6 also means that the binary you inspect was built on your own
machine, for your chip, from your source.

The toolchain, at the pinned commit of dougallj/applegpu:

- `compiler_explorer.py`: takes Metal Shading Language and prints
  instructions. It is the scriptable equivalent of a compiler explorer and the
  entry point for the workflow this chapter describes.
- `disassemble.py`: a 56-line script that you point at extracted GPU machine
  code.
- `applegpu.py`: 6,081 lines of instruction encodings, operand descriptors and
  register formats. This file is where the repository's knowledge lives.
- `assemble.py` goes the other direction; `hwtestbed.py` runs encodings on real
  hardware; `docs.html` is the browsable instruction reference.

The workflow goes beyond reading. Patch instruction bytes
into a compiled shader, run it on the hardware test bed, and diff the GPU's
behavior against the emulator's. That loop answers "what does this undocumented
instruction actually do" by experiment, even though no documentation for it
exists anywhere. It is also how the community established much of what Part 1 reports.

## When to reach for it, and what you find

Reach for it in the order Chapter 9 set: after the timestamps and the roofline
say the kernel should be fast and it is not, and after differential benchmarking
has not found the lever. The usual finds, in order of frequency:

1. **Register spills you did not know you had.** The cliff of Chapter 2 shows up
   here as loads and stores wrapped around arithmetic that should have been
   register-only. This is the single most common finding, and it is why the
   attribute of Chapter 12 and the 16-bit rule of Chapter 4 exist.
2. **Loops that did not unroll the way the source suggests.** The matrix
   multiply of Part 4 asks for its fragment loop and its load loop to become
   straight-line code: the bounds are compile-time constants and the loops carry
   a full-unroll pragma. A compiler can still decline, or unroll partially and
   leave a remainder loop, and when it does the kernel pays loop overhead and
   loses some of the scheduling freedom of Chapter 4. The disassembly is the only
   place you can see which happened.
3. **Instruction-selection surprises.** Expect one surprise every time you
   look; the disassembly is where you look.

For the kernel of Part 4, expect the following: the inner loop should be a short
straight-line sequence of fragment loads and multiply-accumulates with no
memory operations on the accumulators between them. A store inside that
sequence is a suspect, not a verdict: check where it goes and whether the same
value is loaded back later in the loop. An intended output or staging write goes
to a buffer and is never reloaded; a spill goes to compiler-managed backing
storage and comes back. If it is a spill, the levers are the tile size and the
attribute of Chapter 12, with the caveat that chapter recorded: the enlarged
tile spilled with the attribute already present, so the attribute is a hint, not
a cure.

Two practical notes before you run it. The pinned `metal-matmul` kernel does
not compile from its raw source alone; `sync_copy.metal` expects `SW`,
`SIMD_TILE` and `TILE_K` to arrive as preprocessor definitions from the host,
and the applegpu compile helper takes no `-D` option, so write the three values
into a copy of the source before you hand it over, and make sure they match the
configuration you timed. And when the disassembler meets an encoding it does not
know, it prints a failure marker and advances two bytes, so if the unknown
instruction was longer than that, everything after it may be misaligned. Treat
the output after the first unknown encoding as unreliable.

## Two caveats

First, coverage tracks the reverse-engineering effort, not Apple's release
schedule. The newest instructions, the tensor paths of the M5 neural
accelerators in Chapter 29 especially, lag until someone maps them, so a kernel
built on the newest hardware may disassemble into unknown encodings.

Second, everything here is unsupported surface. It is the same platform-risk
trade that the copy-engine story of Chapter 14 documented. The toolchain is a
debugging instrument. Use it to read machine code, and do not build or ship a
product that depends on it.

## The methodology, complete

With this chapter the measurement method of Part 3 is complete. It is stated
here in one place because on this platform it replaces the profiler.

1. Time with command-buffer timestamps, Chapter 9, after the pipeline is built
   and the first runs are discarded, over several trials.
2. Compute achieved operations per second and bytes per second, and compare
   them with ceilings you measured yourself, Chapter 10.
3. If one ratio is near 1, the kernel is at that ceiling as you measured it.
   For the bandwidth ceiling that means fewer bytes. For the compute reference,
   which Chapter 10 took from the best kernel you could find rather than from a
   proven hardware limit, it means fewer operations or a better kernel than the
   reference. If both are low, change one thing at a time and measure again.
4. If the arithmetic says the kernel should be fast and it is not, read the
   machine code, this chapter, and look for spills and loops that did not
   unroll.
5. At every step, report sustained numbers at stated sizes, and report end to
   end next to the kernel number, Chapter 10.

## Check yourself

1. A kernel's time is 4 times its roofline prediction, and changing the tile
   size, the threadgroup size and the precision one at a time moved it by less
   than 10 percent each. What does the chapter say to do, and what are you
   looking for first?
2. Why can the binary you disassemble on your machine differ from the one
   another engineer sees on theirs?
3. Why does the chapter say not to build a product on the applegpu toolchain,
   even though it is the usual way to read the machine code?

### Answers

1. Read the disassembly with `compiler_explorer.py`, and look first for register
   spills: stores of accumulator registers inside the inner loop that are
   loaded back later in the same loop.
2. Because the final compile happens on the device, so the machine code is
   produced by that machine's compiler for that machine's GPU generation,
   Chapter 6. Normally that happens at pipeline creation; the applegpu helper
   triggers the same compile by building a binary archive and serializing it,
   which is how it gets bytes to disassemble.
3. Because it is a reverse-engineered, unsupported interface whose coverage
   lags the hardware, the same risk class as the dead copy intrinsic of Chapter
   14; it is for reading, and a product built on it can break with any release.

One machine's ceilings are now fully described; the next chapter is about what
happens when the model does not fit in one machine.

Next: [Scaling out across machines](27-scaling-out.md)
