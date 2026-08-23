import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://metalworking.vercel.app",
  trailingSlash: "always",
  build: { format: "directory" },
});
