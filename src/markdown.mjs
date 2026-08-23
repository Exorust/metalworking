// Markdown -> HTML with the glossary's link/code conventions.
import { marked } from "marked";
import { REPO } from "./manifest.mjs";

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
