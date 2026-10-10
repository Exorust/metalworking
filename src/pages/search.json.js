import { SECTIONS, BOOK, firstPara, firstProse } from "../manifest.mjs";
import { pageMd, bookMd } from "../content.mjs";

export function GET() {
  const index = [];
  for (const p of BOOK) {
    for (const [file, title] of p.chapters) {
      const md = bookMd(file) ?? "";
      index.push({
        route: `/book/${file}/`,
        title: `${Number(file.slice(0, 2))}. ${title}`,
        section: `Book, part ${p.n}`,
        text: firstProse(md).slice(0, 220),
      });
    }
  }
  for (const s of SECTIONS) {
    for (const [slug, title] of s.pages) {
      const md = pageMd(s.dir, slug);
      index.push({
        route: `/glossary/${s.dir}/${slug}/`,
        title,
        section: `Glossary, ${s.title}`,
        text: firstPara(md).slice(0, 220),
      });
    }
  }
  return new Response(JSON.stringify(index), {
    headers: { "Content-Type": "application/json" },
  });
}
