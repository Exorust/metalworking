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
// Pixel arrow on the same grid language as the letters: 1-cell shaft, stepped head.
const ARROW_GRID = [
  "0000001000",
  "0000001100",
  "0000001110",
  "1111111111",
  "0000001110",
  "0000001100",
  "0000001000",
];
function pixelArrow() {
  const U = 3; // unit cell px
  const rects = [];
  ARROW_GRID.forEach((row, r) => {
    for (let c = 0; c < row.length; c++)
      if (row[c] === "1") rects.push(`<rect x="${c * U}" y="${r * U}" width="${U}" height="${U}"/>`);
  });
  const w = ARROW_GRID[0].length * U, h = ARROW_GRID.length * U;
  return `<svg class="arr" viewBox="0 0 ${w} ${h}" width="${w * 0.9}" height="${h * 0.9}" aria-hidden="true" fill="currentColor">${rects.join("")}</svg>`;
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
  // Focusable code blocks (keyboard users can scroll them; arrow-nav is
  // guarded while they're focused) with comment lines dimmed.
  const escHtml = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  renderer.code = ({ text }) => {
    const body = text.split("\n").map((l) => {
      const t = l.trimStart();
      return t.startsWith("//") || t.startsWith("/*")
        ? `<span class="cmt">${escHtml(l)}</span>` : escHtml(l);
    }).join("\n");
    return `<pre tabindex="0"><code>${body}</code></pre>\n`;
  };
  return marked.parse(md, { renderer, tokenizer });
}

// ---- template ---------------------------------------------------------------
function sidebar(activeRoute) {
  const home = `<div class="toc-section" data-sec="Home"><div class="toc-head"><span>Home</span><button class="fold" aria-label="toggle section">&minus;</button></div><ul><li><a href="/"${activeRoute === "/" ? ' class="active" aria-current="page"' : ""}>README</a></li></ul></div>`;
  const secs = SECTIONS.map(s => {
    const items = s.pages.map(([slug, title, chip]) => {
      const route = `/${s.dir}/${slug}/`;
      const cls = route === activeRoute ? ' class="active" aria-current="page"' : "";
      return `<li><a href="${route}"${cls}>${title}${chip ? ` <span class="chip">${chip}</span>` : ""}</a></li>`;
    }).join("");
    return `<div class="toc-section" data-sec="${s.dir}"><div class="toc-head"><span>${s.title}</span><button class="fold" aria-label="toggle">&minus;</button></div><ul>${items}</ul></div>`;
  }).join("");
  return home + secs;
}

const SITE = "https://metalworking.vercel.app";
const DEFAULT_DESC = "A hyperlinked glossary of Apple Metal GPU performance: the machine, the Metal stack, MLX, techniques, and real kernels.";
const escAttr = (s) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

function page({ route, title, content, prev, next, desc }) {
  const crumb = route === "/" ? "/readme" : route.replace(/\/$/, "");
  const nav = [
    prev ? `<a class="pager prev" href="${prev.route}"><span class="pager-kicker">${pixelArrow()} Previous</span><span class="pager-title">${prev.title}</span></a>` : `<span class="pager-empty"></span>`,
    next ? `<a class="pager next" href="${next.route}"><span class="pager-kicker">Next ${pixelArrow()}</span><span class="pager-title">${next.title}</span></a>` : `<span class="pager-empty"></span>`,
  ].join("");
  const srcPath = route === "/" ? "README.md" : `glossary${route.replace(/\/$/, "")}.md`;
  const pageTitle = title === "README" ? "metalworking — Metal Glossary" : title + " — metalworking";
  const description = escAttr(desc || DEFAULT_DESC);
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${pageTitle}</title>
<meta name="description" content="${description}">
<link rel="canonical" href="${SITE}${route}">
<link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
<meta property="og:type" content="website">
<meta property="og:title" content="${escAttr(pageTitle)}">
<meta property="og:description" content="${description}">
<meta property="og:url" content="${SITE}${route}">
<meta property="og:image" content="${SITE}/assets/banner.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="stylesheet" href="/assets/style.css">
<script>try{document.documentElement.dataset.theme=localStorage.getItem("mw-theme")||"light"}catch(e){}</script>
</head>
<body data-prev="${prev ? prev.route : ""}" data-next="${next ? next.route : ""}">
<a class="skip" href="#main">Skip to content</a>
<div class="frame">
<header class="top">
  <button class="burger" aria-label="Open navigation">&#9776;</button>
  <a class="wordmark" href="/">${pixelLine("M", "px-mark")}metalworking</a>
  <span class="site-title" title="Metal Glossary">${HEADER_ART}</span>
  <nav class="themes" aria-label="theme">
    <button data-theme="terminal">Terminal</button>
    <button data-theme="green">Light green</button>
    <button data-theme="light">Light</button>
    <button data-theme="violet">Violet Midnight</button>
  </nav>
  <button class="searchbox" id="search-open"><svg class="mag" viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><circle cx="6.5" cy="6.5" r="4.5" fill="none" stroke="currentColor" stroke-width="1.6"/><line x1="10" y1="10" x2="14" y2="14" stroke="currentColor" stroke-width="1.6"/></svg> Search <kbd id="search-kbd">&#8984;K</kbd></button>
  <a class="gh-btn" href="${REPO}" target="_blank" rel="noopener">GitHub <span class="ext">&#8599;</span></a>
</header>
<div class="body">
<div class="backdrop" id="nav-backdrop" hidden></div>
<aside class="toc" id="toc"><div class="toc-bar"><div class="toc-title">TABLE OF CONTENTS</div><button class="toc-close" aria-label="Close navigation">&times;</button></div>${sidebar(route)}</aside>
<main class="content" id="main">
  <div class="crumb">${crumb}</div>
  ${content}
  <div class="pagenav">${nav}</div>
  <div class="pagefoot"><a href="${REPO}/blob/main/${srcPath}" target="_blank" rel="noopener">Edit this page on GitHub <span class="ext">&#8599;</span></a></div>
</main>
</div>
</div>
<div class="search-overlay" id="search-overlay" hidden>
  <div class="search-panel" role="dialog" aria-modal="true" aria-label="Search the glossary">
    <input id="search-input" type="text" placeholder="Search the glossary..." autocomplete="off">
    <ul id="search-results"></ul>
    <div class="search-hint"><kbd>&uarr;</kbd><kbd>&darr;</kbd> navigate &nbsp; <kbd>&crarr;</kbd> open &nbsp; <kbd>esc</kbd> close</div>
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
// Azeret Mono (SIL OFL) — self-hosted with its license alongside.
const azeret = join(root, "node_modules/@fontsource/azeret-mono");
for (const w of [400, 500, 700])
  copyFileSync(join(azeret, `files/azeret-mono-latin-${w}-normal.woff2`),
    join(dist, "assets/fonts", `azeret-mono-latin-${w}-normal.woff2`));
copyFileSync(join(azeret, "LICENSE"), join(dist, "assets/fonts", "LICENSE"));

const searchIndex = [];
let count = 0;

// Favicon: the pixel M on a dark tile, accent green.
{
  const rects = [];
  FONT.M.forEach((row, r) => [...row].forEach((c, i) => {
    if (c === "1") rects.push(`<rect x="${6 + i * 4}" y="${6 + r * 4}" width="3.4" height="3.4" fill="#77ef5e"/>`);
  }));
  writeFileSync(join(dist, "assets", "favicon.svg"),
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="4" fill="#0d180a"/>${rects.join("")}</svg>`);
}

// Landing page: README with the banner image swapped for the pixel title.
{
  let md = readFileSync(join(root, "README.md"), "utf8");
  md = md.replace(/^!\[[^\]]*\]\(banner\.png\)\n?/m, "");
  // Self-referential on the site itself; keep it in the GitHub README only.
  md = md.replace(/^\*\*Read it as a website:.*$\n?/m, "");
  const content = TITLE_ART + `<h1 class="visually-hidden">metalworking</h1>` + render(md, null);
  const next = ORDER[1];
  writeFileSync(join(dist, "index.html"),
    page({ route: "/", title: "README", content, prev: null, next, desc: DEFAULT_DESC }));
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
    const firstPara = md.split(/\n\n+/).map(x => x.trim())
      .find(x => x.startsWith("**"))?.replace(/[*\[\]]/g, "").replace(/\(([^)]*)\)/g, "") ?? "";
    writeFileSync(join(out, "index.html"),
      page({ route, title, content, prev: ORDER[i - 1] ?? null, next: ORDER[i + 1] ?? null,
             desc: firstPara.slice(0, 300) }));
    searchIndex.push({ route, title, section: s.title, text: firstPara.slice(0, 220) });
    count++;
  }
}

writeFileSync(join(dist, "search.json"), JSON.stringify(searchIndex));
console.log(`built ${count} pages -> dist/`);
