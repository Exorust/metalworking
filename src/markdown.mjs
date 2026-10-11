// Markdown -> HTML with the glossary's link/code conventions.
import { readFileSync } from "node:fs";
import { marked } from "marked";
import { REPO } from "./manifest.mjs";
import { assetUrl } from "./assets.mjs";
import { themedDiagram, hasDiagram } from "./diagram.mjs";

// fromDir: null for README-level links, "book" for chapters, else a glossary section dir.
function rewriteHref(href, fromDir) {
  if (/^https?:|^mailto:/.test(href)) return { href, external: true };
  const [path, hash = ""] = href.split("#");
  let target = path;
  if (fromDir === null) {                       // README-level links
    if (!target.startsWith("glossary/") && !target.startsWith("book/"))
      return { href: `${REPO}/blob/main/${target}`, external: true }; // SOURCES.md, LICENSE
  } else if (fromDir === "book") {              // chapters link only to chapters
    if (target.startsWith("../")) return { href: `${REPO}/blob/main/${target.replace(/^(\.\.\/)+/, "")}`, external: true };
    target = `book/${target}`;
  } else {
    if (target.startsWith("../../")) return { href: `${REPO}/blob/main/${target.replace(/^(\.\.\/)+/, "")}`, external: true };
    else if (target.startsWith("../")) target = `glossary/${target.slice(3)}`;
    else target = `glossary/${fromDir}/${target}`;
  }
  return { href: `/${target.replace(/\.md$/, "/")}${hash ? "#" + hash : ""}`, external: false };
}

// Pixel size of a GIF or PNG under public/assets/, read from the file header, so the page
// reserves its space before the image arrives (no layout shift).
function imgDims(name) {
  try {
    const b = readFileSync(`public/assets/${name}`);
    if (b.subarray(0, 3).toString() === "GIF") return [b.readUInt16LE(6), b.readUInt16LE(8)];
    if (b[0] === 0x89 && b.subarray(1, 4).toString() === "PNG") return [b.readUInt32BE(16), b.readUInt32BE(20)];
  } catch (e) {}
  return null;
}

export function render(md, fromDir) {
  // Drop the in-file nav lines; the layout provides prev/next.
  md = md.replace(/^Next(?: section)?: \[.*$\n?/gm, "");
  const renderer = new marked.Renderer();
  // This glossary uses "~" for "approximately"; marked's GFM would tokenize
  // ~a ... ~b spans as strikethrough (and break emphasis crossing them).
  const tokenizer = new marked.Tokenizer();
  tokenizer.del = () => undefined;
  renderer.link = function ({ href, tokens }) {
    const text = this.parser.parseInline(tokens);
    const r = rewriteHref(href, fromDir);
    return r.external
      ? `<a href="${r.href}" target="_blank" rel="noopener">${text}<span class="ext" aria-hidden="true">↗</span><span class="visually-hidden"> (opens in a new tab)</span></a>`
      : `<a class="term" href="${r.href}">${text}</a>`;
  };
  renderer.image = ({ href, text }) => {
    const name = href.split("/").pop();
    if (hasDiagram(name)) return themedDiagram(name);
    const d = imgDims(name);
    return `<img src="${assetUrl(name)}" alt="${text}" loading="lazy" decoding="async"${d ? ` width="${d[0]}" height="${d[1]}"` : ""}>`;
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
  // Tables scroll sideways inside a focusable wrapper, so keyboard users can reach clipped columns.
  return marked.parse(md, { renderer, tokenizer })
    .replace(/<table>[\s\S]*?<\/table>/g, (t) => `<div class="table-wrap" tabindex="0">${t}</div>`);
}
