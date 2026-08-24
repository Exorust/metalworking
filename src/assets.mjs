// Content-hashed asset URLs, so a stale cached stylesheet or app.js can never
// pair with fresh HTML (that mismatch silently breaks the JS-driven widgets).
// Contents come through the bundler, not fs: import.meta.url points at the
// compiled chunk during `astro build`, not at this source file.
import { createHash } from "node:crypto";

const files = import.meta.glob("../public/assets/{style.css,app.js}", {
  query: "?raw",
  import: "default",
  eager: true,
});

function v(name) {
  const body = files[`../public/assets/${name}`] ?? "";
  return `/assets/${name}?v=${createHash("sha1").update(body).digest("hex").slice(0, 8)}`;
}

export const STYLE_URL = v("style.css");
export const APP_URL = v("app.js");
