// Inlines diagram SVGs with theme tokens so they recolor with the site.
// Committed SVGs keep baked Terminal hex values for GitHub; each hex below
// maps to a --dg-* variable (fallback = the baked value). Authoring rule:
// new diagrams may only use hexes that appear in MAP (plus #0d180a).
const raw = import.meta.glob(["../memory-hierarchy.svg", "../diagrams/*.svg"],
  { query: "?raw", import: "default", eager: true });

const MAP = [
  ["#24361d", "--dg-div"],
  ["#77ef5e", "--dg-acc"],
  ["#a6cc97", "--dg-soft"],
  ["#12240d", "--dg-core"],
  ["#1c3315", "--dg-regs"],
  ["#d9ead0", "--dg-regs-fg"],
  ["#0f1e0c", "--dg-mem"],
  ["#6f9a62", "--dg-mut"],
  ["#2c4423", "--dg-ghost-line"],
  ["#4a6a3e", "--dg-ghost-fg"],
  ["#2c5423", "--dg-slc"],
  ["#e8c268", "--dg-amber"],
  ["#1a1608", "--dg-on-amber"],
  ["#8a8f7a", "--dg-gray"],
  ["#b9bfa9", "--dg-gray-fg"],
];

const KEYS = Object.fromEntries(
  Object.keys(raw).map((k) => [k.split("/").pop(), k]));

export function hasDiagram(name) {
  return name in KEYS;
}

export function themedDiagram(name = "memory-hierarchy.svg") {
  let svg = raw[KEYS[name]];
  // The background rect first; remaining #0d180a occurrences are dark-on-accent text.
  svg = svg.replace('rx="6" fill="#0d180a"', 'rx="6" fill="var(--dg-bg, #0d180a)"');
  svg = svg.replaceAll("#0d180a", "var(--dg-on-acc, #0d180a)");
  for (const [hex, token] of MAP) svg = svg.replaceAll(hex, `var(${token}, ${hex})`);
  return svg.replace("<svg ", '<svg class="diagram" ');
}
