# The three questions

## Where we are

Parts 4 to 6 built and read the kernels. Part 7 is about what happens when
people try to make real models faster with them, and the record is clear: most
of the measured wins on this platform did not come from writing a better kernel.
They came from deleting work, from moving a model onto a fast path that already
existed, and from cutting the overhead between kernels. This chapter states that
record as three questions to ask, in order, before writing any custom kernel.
Every measured win in the next chapter answers one of them, and nearly every
measured failure skipped them.

## The questions

1. **Can I delete work?**
2. **Can I unlock an existing fast path?**
3. **Can I cut dispatch and synchronization overhead?**

Only if all three come up empty does a custom kernel have decent odds, because
the default kernels, steel and llama.cpp's, are good, and the repeated
experimental result of Chapter 25 is that hand-written replacements come back
slower, at 0.5 to 0.8 times the stock kernel. The order matters: each question is
cheaper to answer than the next, and each one's win is larger than a kernel win
of the same effort tends to be.

## Question 1: delete work

The purest specimen is the sparse-V dequantization skip of Chapter 25: 22.8
percent faster decode at a 32K context from three lines that skip work the kernel
can prove negligible. The same family includes steel attention's causal
handling, Chapter 21, which never visits the tiles above the diagonal rather than
masking them, and vocabulary compaction, which trims the output matrix multiply
when decoding is constrained to a subset of tokens.

The test for a question-1 win is the law of Chapter 3 in its simplest form: did
the change move fewer bytes? Deleting arithmetic on a bandwidth-bound kernel
buys little; deleting the bytes the arithmetic would have needed buys the time
those bytes took. The skip of Chapter 25 deletes bandwidth, which is the deletion
that matters here.

## Question 2: unlock a fast path

The stack is full of dispatch gates: the head-dimension list of Chapter 21, the
sequence-length thresholds of the same gate, the tile-size lookup table of
Chapter 17. Falling off a fast path is silent, or was at the pinned commits, and
the fix is usually a few lines that do not touch a kernel. The measured wins:

- **DFlash**, a port of speculative decoding, reached 3.34 times end to end with
  bit-identical output. Every hand-written custom kernel tried along the way
  lost to stock MLX and was reverted; the real kernel-level win was a
  head-dimension patch that moved the model inside the fused attention gate.
- **MTPLX**, a multi-token prediction project, reached 2.24 times, 28 to 63 tokens
  per second. Stock quantized matrix-vector multiply, Chapter 23, stalls at the
  3 to 6 row shapes that speculative decode produces, between the single-row
  tuning and the tile sizes of the matrix path. About 10 lines of Metal, wider
  simdgroups plus unrolling, bit-exact, fixed it. A textbook question-2 result:
  the kernel existed; its tuning grid had a hole.

The method for a question-2 win is reading, and the files to read are the ones
Chapters 17 and 21 showed: the branch your shape takes, and the condition that
sent it there.

## Question 3: cut dispatch and synchronization overhead

The platform tax of Chapter 7 is real, and the fixes are cheap. DFlash halved
its GPU-to-CPU synchronizations with a single `mx.eval()` per step, Chapter 16.
flash-moe, a mixture-of-experts runner that serves a 209 GB model at 20 tokens
per second in 128 GB of RAM, in pure C, Objective-C and Metal, encodes command
buffers before the GPU needs them and prefetches expert weights during GPU
compute, on the observation that about 27 percent of experts recur from one
token to the next. Its development thread lists 28 failed experiments alongside
the 8 that worked, which is the ratio to expect.

The test for a question-3 win is a count: command buffers per step, dispatches
per step, and waits per step, before and after.

## The discipline the questions encode

The profitable surface on this platform is mostly above the kernel: in dispatch
gates, in graph structure, and in deleted work. A CUDA engineer's instinct is to
go below it first, because on NVIDIA the library kernels are often the thing you
cannot beat and the surrounding code is where you look; here the library kernels
are also hard to beat, and the surrounding code is thinner and more readable, so
the instinct points the wrong way more often. Chapter 25 is what happens when
effort goes below the kernel first, and it is published negative results, which
makes it the most instructive reading on the platform.

The questions also connect back to the arithmetic. Question 1 is Chapter 3's
byte count applied to the whole model. Question 2 is Chapter 17's file reading.
Question 3 is Chapter 7's count of buffers and waits. None of the three needs a
profiler, which is fortunate, since Chapter 9 said there is not one.

## Check yourself

1. A kernel engineer proposes rewriting the attention kernel because the model's
   head dimension is 112 and attention is slow. Which question applies first, and
   what is the likely fix?
2. Speculative decoding sends 4 draft tokens per step and decode throughput drops
   below single-token decode. Which question, and what did the measured case do?
3. A decode loop checks a stopping condition on the GPU's output in Python every
   token. Which question, and what is the cost it pays?

### Answers

1. Question 2: 112 is outside the fused-attention gate of Chapter 21, so the
   model is running the unfused fallback. The likely fix is a head-dimension
   patch that moves it inside the gate, the DFlash win, not a new kernel.
2. Question 2 again: the quantized matrix-vector kernel is tuned for one row and
   the matrix path for many, and 4 rows falls between them. MTPLX fixed it with
   about 10 lines of wider-simdgroup unrolling for a 2.24 times end-to-end gain.
3. Question 3: the Python check forces an evaluation mid-loop, Chapter 16, which
   is a synchronization boundary per token; the fix is to structure the loop for
   one evaluation at the end, the DFlash halving of syncs.

The next chapter is the evidence behind the questions: the measured wins, and the
published failures, with the numbers.

Next: [Wins and failures, measured](25-wins-and-failures-measured.md)
