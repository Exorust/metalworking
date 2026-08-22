// Static site generator for the metalworking glossary.
// Reads README.md + glossary/**/*.md, emits dist/ with clean URLs.
import { marked } from "marked";
import { mkdirSync, readFileSync, writeFileSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "dist");
const REPO = "https://github.com/Exorust/metalworking";

// Reading order = sidebar = prev/next chain. chip: the CUDA-equivalent tag.
const SECTIONS = [
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

const ORDER = [{ route: "/", title: "README", section: "Home" }];
for (const s of SECTIONS)
  for (const [slug, title] of s.pages)
    ORDER.push({ route: `/${s.dir}/${slug}/`, title, section: s.title });

// ---- pixel title (original 5x5 font) ----------------------------------------
const FONT = {
  M: ["10001","11011","10101","10001","10001"],
  E: ["11111","10000","11110","10000","11111"],
  T: ["11111","00100","00100","00100","00100"],
  A: ["01110","10001","11111","10001","10001"],
  L: ["10000","10000","10000","10000","11111"],
  W: ["10001","10001","10101","11011","10001"],
  O: ["01110","10001","10001","10001","01110"],
  R: ["11110","10001","11110","10100","10011"],
  K: ["10001","10010","11100","10010","10001"],
  I: ["11111","00100","00100","00100","11111"],
  N: ["10001","11001","10101","10011","10001"],
  G: ["01111","10000","10111","10001","01110"],
  S: ["01111","10000","01110","00001","11110"],
  Y: ["10001","01010","00100","00100","00100"],
};
function pixelLine(word, cls = "px-art") {
  const P = 8, GAP = 1, LGAP = 6; // px per cell, cell gap, letter gap
  let x = 0; const letters = [];
  for (const ch of word) {
    if (ch === " ") { x += 3 * (P + GAP); continue; }
    const g = FONT[ch];
    const rects = [];
    for (let r = 0; r < 5; r++)
      for (let c = 0; c < 5; c++)
        if (g[r][c] === "1")
          rects.push(`<rect x="${x + c * (P + GAP)}" y="${r * (P + GAP)}" width="${P}" height="${P}"/>`);
    letters.push({ x, body: rects.join("") });
    x += 5 * (P + GAP) + LGAP;
  }
  const w = x - LGAP, h = 5 * (P + GAP) - GAP;
  // One <g> per letter so the landing page can "type" them in sequence.
  const body = letters.map(L =>
    `<g class="px-letter" data-x="${L.x}"><g class="px-shadow" transform="translate(3.5,3.5)">${L.body}</g><g class="px-fill">${L.body}</g></g>`).join("");
  return `<svg class="${cls}" viewBox="-3 -3 ${w + 6} ${h + 6}" role="img" aria-label="${word}" preserveAspectRatio="xMinYMin meet">${body}</svg>`;
}
// Landing composition mirrors banner.png: big METAL, smaller WORKING, tagline.
const TITLE_ART = `<div class="pixel-title">${pixelLine("METAL", "px-art px-big")}${pixelLine("WORKING", "px-art px-small")}<div class="tagline">The craft of making Apple Silicon GPUs go fast.</div></div>`;
const HEADER_ART = pixelLine("METAL GLOSSARY", "px-mini");

// ---- markdown ---------------------------------------------------------------
function rewriteHref(href, fromDir) {
  if (/^https?:|^mailto:/.test(href)) return { href, external: true };
  const [path, hash = ""] = href.split("#");
  let target = path;
  if (fromDir === null) {                       // README-level links
    if (target.startsWith("glossary/")) target = target.slice("glossary/".length);
    else return { href: `${REPO}/blob/main/${target}`, external: true }; // SOURCES.md, LICENSE
  } else {
    if (target.startsWith("../../")) return { href: `${REPO}/blob/main/${target.replace(/^(\.\.\/)+/, "")}`, external: true };
    else if (target.startsWith("../")) target = target.slice(3);
    else target = `${fromDir}/${target}`;
  }
  return { href: `/${target.replace(/\.md$/, "/")}${hash ? "#" + hash : ""}`, external: false };
}

function render(md, fromDir) {
  // Drop the generated-nav source lines; the template provides prev/next.
  md = md.replace(/^Next(?: section)?: \[.*$\n?/gm, "");
  const renderer = new marked.Renderer();
  // This glossary uses "~" for "approximately"; marked's GFM would tokenize
  // ~a ... ~b spans as strikethrough (and break emphasis crossing them).
  // Nothing here wants strikethrough, so turn the del tokenizer off.
  const tokenizer = new marked.Tokenizer();
  tokenizer.del = () => undefined;
  renderer.link = function ({ href, tokens }) {
    const text = this.parser.parseInline(tokens);
    const r = rewriteHref(href, fromDir);
    return r.external
      ? `<a href="${r.href}" target="_blank" rel="noopener">${text}<span class="ext">↗</span></a>`
      : `<a class="term" href="${r.href}">${text}</a>`;
  };
  renderer.image = ({ href, text }) => {
    const name = href.split("/").pop();
    return `<img src="/assets/${name}" alt="${text}">`;
  };
  return marked.parse(md, { renderer, tokenizer });
}

// ---- template ---------------------------------------------------------------
function sidebar(activeRoute) {
  const home = `<div class="toc-section" data-sec="Home"><div class="toc-head"><span>Home</span><button class="fold" aria-label="toggle">&minus;</button></div><ul><li><a href="/" class="${activeRoute === "/" ? "active" : ""}">README</a></li></ul></div>`;
  const secs = SECTIONS.map(s => {
    const items = s.pages.map(([slug, title, chip]) => {
      const route = `/${s.dir}/${slug}/`;
      const cls = route === activeRoute ? ' class="active"' : "";
      return `<li><a href="${route}"${cls}>${title}${chip ? ` <span class="chip">${chip}</span>` : ""}</a></li>`;
    }).join("");
    return `<div class="toc-section" data-sec="${s.dir}"><div class="toc-head"><span>${s.title}</span><button class="fold" aria-label="toggle">&minus;</button></div><ul>${items}</ul></div>`;
  }).join("");
  return home + secs;
}

function page({ route, title, content, prev, next }) {
  const crumb = route === "/" ? "/readme" : route.replace(/\/$/, "");
  const nav = [
    prev ? `<a class="pager prev" href="${prev.route}"><svg class="arr" viewBox="0 0 26 10" width="26" height="10" aria-hidden="true"><path d="M26 5H3M7 1L2 5l5 4" fill="none" stroke="currentColor" stroke-width="1.6"/></svg><span>${prev.title}</span></a>` : "<span></span>",
    next ? `<a class="pager next" href="${next.route}"><span>${next.title}</span><svg class="arr" viewBox="0 0 26 10" width="26" height="10" aria-hidden="true"><path d="M0 5h23M19 1l5 4-5 4" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></a>` : "<span></span>",
  ].join("");
  const srcPath = route === "/" ? "README.md" : `glossary${route.replace(/\/$/, "")}.md`;
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title === "README" ? "metalworking — Metal Glossary" : title + " — metalworking"}</title>
<meta name="description" content="A hyperlinked glossary of Apple Metal GPU performance: the machine, the Metal stack, MLX, techniques, and real kernels.">
<link rel="stylesheet" href="/assets/style.css">
<script>try{document.documentElement.dataset.theme=localStorage.getItem("mw-theme")||"light"}catch(e){}</script>
</head>
<body data-prev="${prev ? prev.route : ""}" data-next="${next ? next.route : ""}">
<div class="frame">
<header class="top">
  <button class="burger" aria-label="menu">&#9776;</button>
  <a class="wordmark" href="/">metalworking</a>
  <span class="site-title" title="Metal Glossary">${HEADER_ART}</span>
  <nav class="themes" aria-label="theme">
    <button data-theme="terminal">Terminal</button>
    <button data-theme="green">Light green</button>
    <button data-theme="light">Light</button>
    <button data-theme="violet">Violet Midnight</button>
  </nav>
  <button class="searchbox" id="search-open"><span class="mag">&#8981;</span> Search <kbd>&#8984;K</kbd></button>
  <a class="gh-btn" href="${REPO}" target="_blank" rel="noopener">GitHub <span class="ext">&#8599;</span></a>
</header>
<div class="body">
<aside class="toc"><div class="toc-title">TABLE OF CONTENTS</div>${sidebar(route)}</aside>
<main class="content">
  <div class="crumb">${crumb}</div>
  ${content}
  <div class="pagenav">${nav}</div>
  <div class="pagefoot"><a href="${REPO}/blob/main/${srcPath}" target="_blank" rel="noopener">Edit this page on GitHub <span class="ext">&#8599;</span></a></div>
</main>
</div>
</div>
<div class="search-overlay" id="search-overlay" hidden>
  <div class="search-panel">
    <input id="search-input" type="text" placeholder="Search the glossary..." autocomplete="off">
    <ul id="search-results"></ul>
  </div>
</div>
<script src="/assets/app.js"></script>
</body></html>`;
}

// ---- emit -------------------------------------------------------------------
mkdirSync(join(dist, "assets", "fonts"), { recursive: true });
for (const f of ["style.css", "app.js"])
  copyFileSync(join(root, "site/assets", f), join(dist, "assets", f));
for (const f of ["banner.png", "gemm-ladder.gif"])
  copyFileSync(join(root, f), join(dist, "assets", f));
// Fira Mono (SIL OFL) — self-hosted with its license alongside.
const fira = join(root, "node_modules/@fontsource/fira-mono");
for (const w of [400, 500, 700])
  copyFileSync(join(fira, `files/fira-mono-latin-${w}-normal.woff2`),
    join(dist, "assets/fonts", `fira-mono-latin-${w}-normal.woff2`));
copyFileSync(join(fira, "LICENSE"), join(dist, "assets/fonts", "LICENSE"));

const searchIndex = [];
let count = 0;

// Landing page: README with the banner image swapped for the pixel title.
{
  let md = readFileSync(join(root, "README.md"), "utf8");
  md = md.replace(/^!\[[^\]]*\]\(banner\.png\)\n?/m, "");
  const content = TITLE_ART + `<h1 class="visually-hidden">metalworking</h1>` + render(md, null);
  const next = ORDER[1];
  writeFileSync(join(dist, "index.html"), page({ route: "/", title: "README", content, prev: null, next }));
  count++;
}

for (const s of SECTIONS) {
  for (const [slug, title] of s.pages) {
    const route = `/${s.dir}/${slug}/`;
    const md = readFileSync(join(root, "glossary", s.dir, `${slug}.md`), "utf8");
    const content = render(md, s.dir);
    const i = ORDER.findIndex(o => o.route === route);
    const out = join(dist, s.dir, slug);
    mkdirSync(out, { recursive: true });
    writeFileSync(join(out, "index.html"),
      page({ route, title, content, prev: ORDER[i - 1] ?? null, next: ORDER[i + 1] ?? null }));
    const firstPara = md.split(/\n\n+/).map(x => x.trim())
      .find(x => x.startsWith("**"))?.replace(/[*\[\]]/g, "").replace(/\(([^)]*)\)/g, "") ?? "";
    searchIndex.push({ route, title, section: s.title, text: firstPara.slice(0, 220) });
    count++;
  }
}

writeFileSync(join(dist, "search.json"), JSON.stringify(searchIndex));
console.log(`built ${count} pages -> dist/`);
