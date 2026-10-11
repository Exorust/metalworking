// Reading order = sidebar = prev/next chain. chip: the CUDA-equivalent tag.
export const REPO = "https://github.com/Exorust/metalworking";
export const SITE = "https://metalworking.vercel.app";
export const DEFAULT_DESC =
  "A hyperlinked glossary of Apple Metal GPU performance: the machine, the Metal stack, MLX, techniques, and real kernels.";

export const SECTIONS = [
  { dir: "machine", title: "The Machine", pages: [
    ["nvidia-to-metal", "NVIDIA to Metal", ""],
    ["gpu-core", "GPU Core", "SM"],
    ["simdgroup", "Simdgroup", "warp"],
    ["registers", "Registers", ""],
    ["threadgroup-memory", "Threadgroup Memory", "smem"],
    ["unified-memory", "Unified Memory", ""],
    ["occupancy", "Occupancy", ""],
    ["f16", "F16", "fp16"],
    ["special-paths", "Special Paths", ""],
    ["amx", "AMX", ""],
    ["neural-accelerators", "Neural Accelerators", "Tensor Core"],
  ]},
  { dir: "metal", title: "Metal, the Stack", pages: [
    ["metal-the-api", "Metal, the API", "CUDA"],
    ["msl", "MSL", "CUDA C++"],
    ["dispatch-geometry", "Dispatch Geometry", "grid"],
    ["compilation-pipeline", "Compilation Pipeline", "PTX"],
    ["function-constants", "Function Constants", ""],
    ["command-buffers", "Command Buffers", "stream"],
    ["synchronization", "Synchronization", ""],
    ["simdgroup-matrix", "simdgroup_matrix", "wmma"],
    ["simdgroup-async-copy", "simdgroup_async_copy", "cp.async"],
    ["mtltensor-and-mpp", "MTLTensor and MPP", "TMA"],
    ["mps", "MPS", "cuBLAS"],
    ["profiling", "Profiling", "Nsight"],
    ["disassembly", "Disassembly", "SASS"],
  ]},
  { dir: "mlx", title: "MLX", pages: [
    ["mlx-overview", "MLX, an Overview", ""],
    ["lazy-evaluation", "Lazy Evaluation", ""],
    ["how-an-op-becomes-a-kernel", "How an Op Becomes a Kernel", ""],
    ["steel", "Steel", "CUTLASS"],
    ["mx-fast", "mx.fast", ""],
    ["quantization", "Quantization", ""],
    ["mx-compile", "mx.compile", ""],
    ["distributed", "Distributed", "NCCL"],
  ]},
  { dir: "techniques", title: "Techniques", pages: [
    ["arithmetic-intensity", "Arithmetic Intensity", ""],
    ["roofline", "Roofline", ""],
    ["tiling", "Tiling", ""],
    ["cooperative-load", "Cooperative Load", ""],
    ["register-blocking", "Register Blocking", ""],
    ["double-buffering", "Double Buffering", ""],
    ["fusion-and-epilogues", "Fusion and Epilogues", ""],
    ["online-softmax", "Online Softmax", ""],
    ["flash-attention", "Flash Attention", ""],
    ["kv-cache", "The KV Cache", ""],
    ["decode-vs-prefill", "Decode vs Prefill", ""],
  ]},
  { dir: "kernels", title: "Kernels", pages: [
    ["gemm-tiled", "The Tiled GEMM", ""],
    ["gemm-double-buffered", "The Double-Buffered GEMM", ""],
    ["gemm-async-ghost", "The Async-Copy Ghost", ""],
    ["steel-blockloader", "Steel's BlockLoader", ""],
    ["steel-blockmma", "Steel's BlockMMA", ""],
    ["steel-gemm-fused", "The Fused GEMM Kernel", ""],
    ["steel-attention", "Steel Attention", ""],
    ["mfa-codegen", "metal-flash-attention", ""],
    ["llamacpp-attention", "llama.cpp Attention", ""],
    ["nax-gemm", "The NAX GEMM", ""],
  ]},
  { dir: "war-stories", title: "War Stories", pages: [
    ["three-questions", "The Three Questions", ""],
    ["sparse-v", "Sparse-V", ""],
    ["the-failures", "The Failures", ""],
    ["cheap-tricks", "Cheap Tricks", ""],
  ]},
];

// The book: one reading order, parts only here, flat files under book/.
// Each chapter: [file stem, title, one line on what the reader can do after it].
export const BOOK = [
  { n: 1, title: "The machine", question: "What is different about this chip?", chapters: [
    ["01-two-hierarchies", "Two hierarchies: what you launch and what runs it", "Apple uses one word, "core", for a different level than NVIDIA does, and that one word causes a lot of the early confusion."],
    ["02-the-cores-memories", "The core's memories: 208 KB of registers, 32 KB of scratch", "208 KB of registers per core and 32 KB of threadgroup memory, which is why Apple kernels keep their working set in registers and why a spill costs 10 times, not 10 percent."],
    ["03-one-memory-pool-one-law", "One memory pool and the one law: bandwidth", "One memory pool, no transfers, and one law: most kernels are decided by bandwidth before they are written."],
    ["04-keeping-the-core-busy", "Keeping the core busy: occupancy, stalls, and F16", "Occupancy here is resource arithmetic, and F16 wins for stall and register reasons, not for throughput."],
    ["05-the-two-asymmetries", "The two asymmetries: fast exp2, emulated float atomics", "A fast exp2 and float atomics that are emulated: two hardware paths that reshape kernels written for NVIDIA."],
  ]},
  { n: 2, title: "Running a kernel", question: "What is the least Metal you need?", chapters: [
    ["06-from-source-to-pipeline-state", "From source to pipeline state", "The final compile happens on the device at pipeline creation, so the machine code is made on your machine for your chip."],
    ["07-launching-batching-waiting", "Launching, batching, and waiting", "A dispatch is cheap and a wait is not; inference converged on one command buffer per forward pass for a reason."],
    ["08-one-kernel-many-shapes", "One kernel, many shapes", "Function constants turn one shipped kernel source into a variant per case, specialized at run time for the shape at hand."],
  ]},
  { n: 3, title: "Measuring before optimizing", question: "How do you know a kernel is slow?", chapters: [
    ["09-the-baseline-and-the-stopwatch", "The baseline and the stopwatch", "MPS multiplies 4096 by 4096 at 11.7 TFLOPS on an M5 Max; the stopwatch is GPU timestamps, and the first runs are thrown away."],
    ["10-build-your-own-roofline", "Build your own roofline", "The compute roof is measured, not quoted from a spec sheet: 13.5 TFLOPS and a ridge of about 26 FLOPs per byte on the M5 Max."],
  ]},
  { n: 4, title: "One matrix multiply", question: "How does a 144-line kernel beat the library?", chapters: [
    ["11-tiling-and-the-cooperative-load", "Tiling and the cooperative load", "A tile sized for 32 KB of threadgroup memory and a coalesced cooperative load: the first two techniques of the 144-line kernel."],
    ["12-simdgroup-matrix-and-register-blocking", "simdgroup_matrix and register blocking", "The kernel beats MPS by about 15 percent, and the whole win is keeping the accumulator in registers across the K loop."],
    ["13-reading-the-whole-kernel", "Reading the whole kernel", "sync_copy.metal, read end to end, with every one of its lines accounted for."],
    ["14-double-buffering-and-the-copy-engine-that-died", "Double buffering, and the copy engine that died", "The copy intrinsic exists in the headers and is dead on the hardware, so double buffering is yours to do, and it does not always win."],
    ["15-fusion-and-epilogues", "Fusion and epilogues", "The bias, the activation and the scaling around a matmul fold into its one store, and each one folded is a pass over memory you no longer pay."],
  ]},
  { n: 5, title: "The same GEMM in production", question: "What does MLX do with your matmul?", chapters: [
    ["16-arrays-that-record", "Arrays that record: MLX and lazy evaluation", "An MLX array records work instead of doing it, and one mx.eval per step is the difference between a fast loop and a slow one."],
    ["17-how-an-op-becomes-a-kernel", "How an op becomes a kernel", "mx.matmul walks a dispatch tree to a precompiled steel kernel; on an M5 Max in 32-bit it lands on a NAX tile with TF32 on by default."],
    ["18-steel-the-gemm-as-components", "Steel: the GEMM as components", "The hand kernel's three parts, loader, multiply and epilogue, are steel's three components: BlockLoader, BlockMMA and the epilogue."],
    ["19-mx-compile", "mx.compile: fusing the long tail", "mx.compile traces the lazy graph once and fuses the elementwise chains around the matmul; the matmul itself it leaves alone."],
  ]},
  { n: 6, title: "Attention and inference", question: "Why is decode slow, and what can be done?", chapters: [
    ["20-online-softmax-and-flash-attention", "Online softmax and flash attention", "The score tile never touches memory: online softmax corrects a running maximum so attention runs in one pass over the keys."],
    ["21-three-attention-kernels", "Three attention kernels, one algorithm", "Three production attention kernels, one algorithm, three design bets: steel's, metal-flash-attention's generated kernels, and llama.cpp's enumerated ones."],
    ["22-the-kv-cache-decode-and-prefill", "The KV cache, decode, and prefill", "A 32K-token cache is about 4.29 GB, and reading it once per token at 495 GB/s is what makes decode a byte-streaming problem."],
    ["23-quantization", "Quantization as a bandwidth optimization", "Four-bit weights stream about 3.5 times fewer bytes per token, and the QMV kernel multiplies against them without expanding the matrix."],
  ]},
  { n: 7, title: "Working on real models", question: "Where do wins come from?", chapters: [
    ["24-the-three-questions", "The three questions", "Three questions, in order, before any custom kernel: can you delete work, can you unlock an existing fast path, can you cut dispatch and synchronization overhead."],
    ["25-wins-and-failures-measured", "Wins and failures, measured", "A ledger of 14 custom kernels: 2 isolated wins, 11 losses or neutral, 1 with no lever, and the loss modes the failures share."],
    ["26-when-it-is-still-slow", "When it is still slow: reading the machine code", "Apple ships no disassembler; applegpu is how you read the machine code, and register spills are the first thing you look for."],
    ["27-scaling-out", "Scaling out across machines", "Two Mac Studios hold a trillion-parameter model over Thunderbolt 5 RDMA, and whether a split ever wins is one inequality."],
  ]},
  { n: 8, title: "The new hardware", question: "What changes on M5?", chapters: [
    ["28-three-matrix-engines", "Three matrix engines on one die", "Three matrix engines on one die, AMX, the GPU and the Neural Engine, and Metal talks to only one of them."],
    ["29-tensor-ops-and-the-nax-gemm", "Tensor ops and the NAX GEMM", "A matmul2d descriptor hands the multiply to the M5 matrix unit; the tile, the accumulator grid and the lane layout are still yours."],
  ]},
];

export const GLOSSARY_ORDER = [{ route: "/glossary/", title: "Glossary", section: "Glossary" }];
for (const s of SECTIONS)
  for (const [slug, title] of s.pages)
    GLOSSARY_ORDER.push({ route: `/glossary/${s.dir}/${slug}/`, title, section: s.title });

export const BOOK_ORDER = [{ route: "/book/", title: "Contents", section: "Book" }];
for (const p of BOOK)
  for (const [file, title] of p.chapters)
    BOOK_ORDER.push({ route: `/book/${file}/`, title, section: `Part ${p.n}` });

// The definition sentence that opens each glossary entry (bolded first paragraph).
export function firstPara(md) {
  return md.split(/\n\n+/).map((x) => x.trim())
    .find((x) => x.startsWith("**"))
    ?.replace(/[*\[\]]/g, "").replace(/\(([^)]*)\)/g, "") ?? "";
}

// The first prose paragraph of a chapter (skips headings, images, captions).
export function firstProse(md) {
  return md.split(/\n\n+/).map((x) => x.trim())
    .find((x) => x && !/^[#!*`|<-]/.test(x) && !/^\d+\./.test(x))
    ?.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/[*`]/g, "") ?? "";
}
