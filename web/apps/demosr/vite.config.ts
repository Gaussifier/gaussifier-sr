import { defineConfig } from "vite";

// As in apps/demo: onnxruntime-web stays outside the bundle (index.html maps it to
// ./ort/ort.webgpu.min.mjs through an import map; scripts/prepare-assets.mjs copies the runtime),
// and every path is relative, so the built site works from any directory (it is served at /demosr/).
export default defineConfig({
  base: "./",
  build: { outDir: "dist", target: "es2022", sourcemap: false, rollupOptions: { external: ["onnxruntime-web/webgpu", "onnxruntime-web"] } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
