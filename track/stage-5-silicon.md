# Stage 5 — Down to the silicon

**Needs:** `code/applegpu`, `code/amx` &nbsp;·&nbsp; **Time:** 2–3 days; optional but addictive

Sometimes the profiler isn't enough and you need to know what the compiler actually
emitted. This stage is the toolbox for that: a reverse-engineered ISA disassembler for
the GPU, and the definitive documentation of AMX — the undocumented CPU matrix
coprocessor that's the other half of the Apple matmul story (it's how MPS/BNNS get their
small-matrix throughput).

## The GPU ISA — dougallj/applegpu

Apple G13 (M1-family) instruction set, reverse engineered:

- `code/applegpu/disassemble.py` — the entry point, 56 lines: `disassemble(code)` at
  line 8, CLI at line 47. Point it at extracted AIR/GPU machine code.
- `code/applegpu/applegpu.py` — 6081 lines: instruction encodings, operand descriptors,
  register/immediate formats. This file *is* the knowledge in the repo.
- `code/applegpu/compiler_explorer.py` — Metal source in, ISA out: your
  godbolt-for-Metal. `assemble.py` is the reverse direction; `hwtest.py`/`hwtestbed.py`
  verify encodings on real hardware; `docs.html` is the browsable instruction reference.

The killer workflow: patch instruction bytes into a compiled shader and diff GPU behavior
against the emulator — you can empirically answer "what does this instruction actually
do" without any documentation existing.

## The AMX coprocessor — corsix/amx

Not the GPU, but you'll meet it whenever small-matrix CPU throughput seems impossible:

- `code/amx/README.md` — the overview: a 32×32 MAC grid, X/Y register pools, an
  outer-product-per-instruction model, and per-generation deltas (M1→M2 adds bf16;
  M2→M3 adds new load/int modes; M3→M4 changes extraction offset handling).
- `code/amx/emulate.h` + `aarch64.h` — a full software emulator: the *executable
  specification*. Per-instruction `.c`/`.md` pairs (`matfp`, `matint`, `vecfp`,
  `vecint`, `extr`, `ldst`, `genlut`) document and test each op.
- `code/amx/perf.c` / `perf_table.py` — measured achieved throughput.

## Done when

- You've run a stage-2 kernel through `compiler_explorer.py` and read the emitted ISA.
- You found one thing the compiler did that you didn't expect (unrolling, instruction
  selection, register allocation) — there's always one.
- You can say in one sentence what AMX is and why it isn't the GPU.

**Next:** [Stage 6 — War stories](stage-6-war-stories.md)
