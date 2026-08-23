// Raw markdown, resolved by the bundler at compile time (immune to cwd/chunk paths).
const mdFiles = import.meta.glob("../glossary/*/*.md", { query: "?raw", import: "default", eager: true });
const readme = import.meta.glob("../README.md", { query: "?raw", import: "default", eager: true });

export function pageMd(dir, slug) {
  return mdFiles[`../glossary/${dir}/${slug}.md`];
}
export const README = readme["../README.md"];
