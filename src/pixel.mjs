// Original 5x5 pixel font, title art, and arrow glyphs.
export const FONT = {
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

export function pixelLine(word, cls = "px-art") {
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
export function pixelArrow() {
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
export const TITLE_ART =
  `<div class="pixel-title">${pixelLine("METAL", "px-art px-big")}${pixelLine("WORKING", "px-art px-small")}` +
  `<div class="tagline">The craft of making Apple Silicon GPUs go fast.</div></div>`;
export const MARK = pixelLine("M", "px-mark");
