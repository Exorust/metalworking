# Disassembly

**When arithmetic says a kernel's performance is impossible, the last resort
is reading what the compiler actually emitted. Apple ships no tool for that;
the reverse-engineered [applegpu](https://github.com/dougallj/applegpu)
toolchain is how it's done.**

CUDA equivalent: `cuobjdump` / `nvdisasm` reading SASS, except there the
vendor documents the tool and half the ISA. Here the ISA (Apple G13, the
M1 family, with later-generation coverage growing) was reverse engineered by
the community, and the [compilation pipeline](compilation-pipeline.md)'s
on-device final compile means the binary you inspect was built on your own
machine.

The toolchain, at the pinned commit
([applegpu @ `4c5bae61`](https://github.com/dougallj/applegpu/tree/4c5bae61086b8067231120c98b4756d7696d399c)):

- **`compiler_explorer.py`**: MSL in, ISA out. Your Godbolt for Metal, and
  the entry point for the workflow this page exists to name.
- **`disassemble.py`**: 56 lines; point it at extracted GPU machine code.
- **`applegpu.py`**: 6,081 lines of instruction encodings, operand
  descriptors, and register formats. This file *is* the knowledge in the
  repo.
- **`assemble.py`** goes the other direction; **`hwtestbed.py`** runs
  encodings on real hardware; **`docs.html`** is the browsable instruction
  reference.

The workflow worth internalizing goes beyond reading: **patch instruction
bytes into a compiled shader, run it, and diff the GPU's behavior against the
emulator's.** That loop answers "what does this undocumented instruction
actually do" empirically, with no documentation existing anywhere. It is how
the community established most of what the
[machine section](../machine/gpu-core.md) reports.

When to reach for it, per the [profiling methodology](profiling.md): after
timestamps and the [roofline](../techniques/roofline.md) say the kernel
should be fast and it isn't, and differential benchmarking hasn't found the
lever. The usual finds, in order of frequency: [register
spills](../machine/registers.md) you didn't know you had, loops that didn't
unroll the way the source suggests, and instruction selection surprises
(the [compilation-pipeline page](compilation-pipeline.md)'s "expect one
surprise every time you look" is this page's summary).

Two honest caveats. First, coverage tracks the reverse-engineering effort,
not Apple's release schedule: newest-generation instructions (the
[NAX](../machine/neural-accelerators.md) tensor paths especially) lag until
someone maps them. Second, everything here is unsupported surface, the same
[platform-risk trade](../kernels/gemm-async-ghost.md) the async-copy story
documents; it's a debugging instrument, not something to build a product on.

Next section: [MLX](../mlx/mlx-overview.md)
