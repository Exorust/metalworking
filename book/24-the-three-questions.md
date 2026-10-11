# The three questions

## Where we are

You have spent Parts 4 to 6 building and reading the kernels. Now you want a
real model to run faster, and Part 7 is about what happens when people try. The
record shows that most of the measured wins on this platform did not come from
writing a better kernel. They came from deleting work, from moving a model onto a
fast path that already existed, and from cutting the overhead between kernels.

That record fits into three questions, asked in order, before you write any
custom kernel. Every measured win in the next chapter answers one of them, and
nearly every measured failure skipped them.

## The questions

When a model is slow, what do you try before writing a kernel? In order:

1. Can you delete work?
2. Can you move onto an existing fast path?
3. Can you cut dispatch and synchronization overhead?

A custom kernel has decent odds only if all three come up empty. The default
kernels, steel and llama.cpp's, are good, and the repeated experimental result of Chapter 25 is that hand-written replacements come back
slower, at 0.5 to 0.8 times the stock kernel. The order matters. Each question is
cheaper to answer than the next, and in the measured cases below their wins were
larger than the kernel rewrites tried alongside them.

## Question 1: delete work

What does deleting work look like when the model is fixed? The clearest example
is the sparse-V dequantization skip of Chapter 25: 22.8
percent faster decode at a 32K context from three lines that skip positions
whose attention weight is below a threshold, an approximation checked against
perplexity. The same family includes steel attention's causal
handling, Chapter 21, which never visits the tiles above the diagonal rather than
masking them, and vocabulary compaction, which trims the output matrix multiply
when decoding is constrained to a subset of tokens.

You test a question-1 win with the law of Chapter 3 in its simplest form: did
the change move fewer bytes? Deleting arithmetic on a bandwidth-bound kernel
buys little. Deleting the bytes the arithmetic would have needed buys up to the
time those bytes took, if the achieved bandwidth holds and the skip itself is
cheap. The skip of Chapter 25 deletes bandwidth, which is the kind of
deletion that saves time on a bandwidth-bound kernel.

## Question 2: move onto an existing fast path

Your model may already have a fast kernel and never reach it. The stack is full
of dispatch gates: the head-dimension list of Chapter 21, the
sequence-length thresholds of the same gate, the tile-size lookup table of
Chapter 17. Falling off a fast path is silent, or was at the pinned commits, and
the fix is often a few lines, at the dispatch or inside a kernel's tuning rather
than a new kernel.

DFlash, a port of speculative decoding, reached 3.34 times end to end with output
identical to the greedy baseline. Every hand-written custom kernel tried along
the way lost to stock MLX and was reverted; the real kernel-level win was a
head-dimension patch that moved the model inside the fused attention gate.

MTPLX, a multi-token prediction project, reached 2.24 times, 28 to 63 tokens per
second as rounded in its report. Stock quantized matrix-vector multiply, Chapter
23, stalls at the 3 to 6 row shapes that speculative decode produces, between the
single-row tuning and the tile sizes of the matrix path. About 10 lines of Metal
fixed it with wider simdgroup tiles (a simdgroup is still 32 lanes) plus
unrolling, and the output stayed bit-exact. You already had the kernel; its
tuning grid had a hole, the typical question-2 result.

The method for a question-2 win is reading, and the files to read are the ones
Chapters 17 and 21 showed: the branch your shape takes, and the condition that
sent it there.

## Question 3: cut dispatch and synchronization overhead

If the kernels are already the right ones, the time can still go between them.
The platform tax of Chapter 7 is real, but the fixes are cheap. DFlash halved
its GPU-to-CPU synchronizations with a single `mx.eval()` per step, Chapter 16.
flash-moe, a mixture-of-experts runner that serves a 209 GB model at 20 tokens
per second in 128 GB of RAM, in pure C, Objective-C and Metal, encodes command
buffers before the GPU needs them and prefetches expert weights during GPU
compute, on the observation that about 27 percent of experts recur from one
token to the next. Its development thread lists 28 failed experiments alongside
the 8 that worked, a realistic ratio.

You test a question-3 win by starting with a count, command buffers per step,
dispatches per step and waits per step, before and after. You end with a
measurement of exposed encoding time and GPU idle time. Counts alone are
not a result.

## The discipline the questions encode

Why put the kernel last? The profitable surface on this platform is mostly above
the kernel: in dispatch gates, in graph structure, and in deleted work. Your
instinct from CUDA is to write the kernel, because custom kernels routinely win on
NVIDIA away from the shapes the library tuned for. Here the library kernels have held up against
hand-written replacements, and the surrounding dispatch code is thin and
readable, so the profitable moves sit above the kernel more often than that
instinct expects. Chapter 25 is what happens when
effort goes below the kernel first, and because it is a set of published negative
results, it is the most instructive reading on the platform.

The questions also connect back to the arithmetic. Question 1 is Chapter 3's
byte count applied to the whole model, question 2 is Chapter 17's file reading,
and question 3 is Chapter 7's count of buffers and waits. None of the three needs a
profiler, which matters because Chapter 9 said the one that exists cannot run in
a feedback loop.

## Check yourself

1. A colleague proposes rewriting the attention kernel because the model's
   head dimension is 112 and attention is slow. Which question applies first, and
   what is the likely fix?
2. Speculative decoding sends 4 draft tokens per step and decode throughput drops
   below single-token decode. Which question, and what did the measured case do?
3. A decode loop checks a stopping condition in Python in the middle of each
   step, on output the GPU has not finished, and records more work after it.
   Which question, and what is the cost it pays?

### Answers

1. Question 2: 112 is outside the fused-attention gate of Chapter 21, so the
   model is running the unfused fallback. The likely fix is at the dispatch and
   shape level, the head-dimension patch that was DFlash's win, before any new
   kernel; question 1 was skipped here because attention's work is already
   fixed by the model.
2. Question 2 again: the quantized matrix-vector kernel is tuned for one row and
   the matrix path for many, and 4 rows falls between them. MTPLX fixed it with
   about 10 lines of wider-simdgroup unrolling for a 2.24 times end-to-end gain.
3. Question 3: the Python check forces an evaluation of the pending output
   mid-step, Chapter 16, and the work recorded after it forces another at the
   end, two synchronization boundaries per token; the fix is to read the
   stopping value at the end of the step, one evaluation, the DFlash halving of
   syncs.

The next chapter is the evidence behind the questions: the measured wins, and the
published failures, with the numbers.

Next: [Wins and failures, measured](25-wins-and-failures-measured.md)
