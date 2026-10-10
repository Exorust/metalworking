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
    ["01-two-hierarchies", "Two hierarchies: what you launch and what runs it", "Name every level of a launch and of the chip, and say which runs on which."],
    ["02-the-cores-memories", "The core's memories: 208 KB of registers, 32 KB of scratch", "Explain why Apple kernels keep their working set in registers."],
    ["03-one-memory-pool-one-law", "One memory pool and the one law: bandwidth", "Compute a ridge point and say whether a kernel is bandwidth bound."],
    ["04-keeping-the-core-busy", "Keeping the core busy: occupancy, stalls, and F16", "Reason about occupancy from resource arithmetic, and say why F16 wins."],
    ["05-the-two-asymmetries", "The two asymmetries: fast exp2, emulated float atomics", "Recognize the two hardware paths that reshape kernels here."],
  ]},
  { n: 2, title: "Running a kernel", question: "What is the least Metal you need?", chapters: [
    ["06-from-source-to-pipeline-state", "From source to pipeline state", "Read a Metal kernel signature and follow it from source to a pipeline state."],
    ["07-launching-batching-waiting", "Launching, batching, and waiting", "Encode a dispatch, batch many into one command buffer, and wait as rarely as possible."],
    ["08-one-kernel-many-shapes", "One kernel, many shapes", "Specialize one shipped kernel into many variants with function constants."],
  ]},
  { n: 3, title: "Measuring before optimizing", question: "How do you know a kernel is slow?", chapters: [
    ["09-the-baseline-and-the-stopwatch", "The baseline and the stopwatch", "Time a kernel against MPS with GPU timestamps and trust the number."],
    ["10-build-your-own-roofline", "Build your own roofline", "Measure bandwidth and compute ceilings and place a kernel on the chart."],
  ]},
  { n: 4, title: "One matrix multiply", question: "How does a 140-line kernel beat the library?", chapters: [
    ["11-tiling-and-the-cooperative-load", "Tiling and the cooperative load", "Size a tile for this platform and stage it with a coalesced cooperative load."],
    ["12-simdgroup-matrix-and-register-blocking", "simdgroup_matrix and register blocking", "Hold the output in registers across the whole K loop without spilling."],
    ["13-reading-the-whole-kernel", "Reading the whole kernel", "Read sync_copy.metal end to end and account for every line."],
    ["14-double-buffering-and-the-copy-engine-that-died", "Double buffering, and the copy engine that died", "Say when double buffering wins, when it loses, and why there is no DMA."],
    ["15-fusion-and-epilogues", "Fusion and epilogues", "Fold the operations around a matmul into its store."],
  ]},
  { n: 5, title: "The same GEMM in production", question: "What does MLX do with your matmul?", chapters: [
    ["16-arrays-that-record", "Arrays that record: MLX and lazy evaluation", "Explain what mx.eval does and why one eval per step matters."],
    ["17-how-an-op-becomes-a-kernel", "How an op becomes a kernel", "Trace mx.matmul to the kernel variant and tile sizes it dispatches."],
    ["18-steel-the-gemm-as-components", "Steel: the GEMM as components", "Map the hand kernel's three parts onto BlockLoader, BlockMMA, and the epilogue."],
    ["19-mx-compile", "mx.compile: fusing the long tail", "Say what mx.compile fuses, what it does not, and when to reach for it."],
  ]},
  { n: 6, title: "Attention and inference", question: "Why is decode slow, and what can be done?", chapters: [
    ["20-online-softmax-and-flash-attention", "Online softmax and flash attention", "Re-derive the online softmax correction and explain why the score tile never touches memory."],
    ["21-three-attention-kernels", "Three attention kernels, one algorithm", "Tell the three production attention kernels apart by their design bets."],
    ["22-the-kv-cache-decode-and-prefill", "The KV cache, decode, and prefill", "Convert a cache size into decode latency and explain the two workloads."],
    ["23-quantization", "Quantization as a bandwidth optimization", "Explain why 4-bit weights speed up decode and what the QMV kernel does."],
  ]},
  { n: 7, title: "Working on real models", question: "Where do wins come from?", chapters: [
    ["24-the-three-questions", "The three questions", "Ask the three questions, in order, before writing a kernel."],
    ["25-wins-and-failures-measured", "Wins and failures, measured", "Recognize the loss modes of custom kernels and the shape of a real win."],
    ["26-when-it-is-still-slow", "When it is still slow: reading the machine code", "Disassemble a kernel with applegpu and know what to look for."],
    ["27-scaling-out", "Scaling out across machines", "Say when a Thunderbolt cluster wins and what it costs."],
  ]},
  { n: 8, title: "The new hardware", question: "What changes on M5?", chapters: [
    ["28-three-matrix-engines", "Three matrix engines on one die", "Place AMX, the GPU, and the neural accelerators, and say which API reaches each."],
    ["29-tensor-ops-and-the-nax-gemm", "Tensor ops and the NAX GEMM", "Read a Metal 4 tensor-op GEMM and say what the hardware now places for you."],
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
