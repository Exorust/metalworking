import { SECTIONS, firstPara } from "../manifest.mjs";
import { pageMd } from "../content.mjs";

export function GET() {
  const index = [];
  for (const s of SECTIONS) {
    for (const [slug, title] of s.pages) {
      const md = pageMd(s.dir, slug);
      index.push({
        route: `/${s.dir}/${slug}/`,
        title,
        section: s.title,
        text: firstPara(md).slice(0, 220),
      });
    }
  }
  return new Response(JSON.stringify(index), {
    headers: { "Content-Type": "application/json" },
  });
}
