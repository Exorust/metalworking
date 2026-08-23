// Reading order = sidebar = prev/next chain. chip: the CUDA-equivalent tag.
export const REPO = "https://github.com/Exorust/metalworking";
export const SITE = "https://metalworking.vercel.app";
export const DEFAULT_DESC =
  "A hyperlinked glossary of Apple Metal GPU performance: the machine, the Metal stack, MLX, techniques, and real kernels.";

export const SECTIONS = [
  { dir: "machine", title: "The Machine", pages: [
    ["gpu-core", "GPU Core", "SM"],
    ["simdgroup", "Simdgroup", "warp"],
    ["registers", "Registers", ""],
    ["threadgroup-memory", "Threadgroup Memory", "smem"],
    ["unified-memory", "Unified Memory", ""],
    ["occupancy", "Occupancy", ""],
    ["f16", "F16", "fp16"],
    ["special-paths", "Special Paths", ""],
    ["amx", "AMX", ""],
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
    ["mps", "MPS", "cuBLAS"],
    ["profiling", "Profiling", "Nsight"],
  ]},
  { dir: "mlx", title: "MLX", pages: [
    ["mlx-overview", "MLX, an Overview", ""],
    ["lazy-evaluation", "Lazy Evaluation", ""],
    ["how-an-op-becomes-a-kernel", "How an Op Becomes a Kernel", ""],
    ["steel", "Steel", "CUTLASS"],
    ["mx-fast", "mx.fast", ""],
    ["quantization", "Quantization", ""],
    ["mx-compile", "mx.compile", ""],
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
  ]},
  { dir: "war-stories", title: "War Stories", pages: [
    ["three-questions", "The Three Questions", ""],
    ["sparse-v", "Sparse-V", ""],
    ["the-failures", "The Failures", ""],
    ["cheap-tricks", "Cheap Tricks", ""],
  ]},
];

export const ORDER = [{ route: "/", title: "README", section: "Home" }];
for (const s of SECTIONS)
  for (const [slug, title] of s.pages)
    ORDER.push({ route: `/${s.dir}/${slug}/`, title, section: s.title });

// The definition sentence that opens each page (bolded first paragraph).
export function firstPara(md) {
  return md.split(/\n\n+/).map((x) => x.trim())
    .find((x) => x.startsWith("**"))
    ?.replace(/[*\[\]]/g, "").replace(/\(([^)]*)\)/g, "") ?? "";
}
