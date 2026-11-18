// Copy the runtime assets the page loads into public/ (all gitignored):
//   public/ort/              ONNX Runtime Web: the minified WebGPU build and the one wasm build it loads
//   public/oversampler/      prebuilt Emscripten oversampler glue + wasm
//   public/samples/kodak/    the Kodak images and thumbnails committed under apps/demo
// public/models/ is written by scripts/export_web_bundle.py. The page loads no source maps, so
// the comment that points the minified ORT build at its map is dropped.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const app = path.resolve(here, "..");
const web = path.resolve(app, "../..");
const copy = (src, dst, transform) => {
  if (!fs.existsSync(src)) { console.warn(`[prepare-assets] missing ${src}`); return; }
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  if (transform) fs.writeFileSync(dst, transform(fs.readFileSync(src, "utf8"))); else fs.copyFileSync(src, dst);
};
const dropSourceMap = (text) => text.replace(/\n?\/\/# sourceMappingURL=\S*\s*$/, "\n");

const ortDist = path.join(web, "node_modules/onnxruntime-web/dist");
const ortEntry = path.join(ortDist, "ort.webgpu.min.mjs");
// The WebGPU build names the wasm glue it imports (the asyncify build in ORT 1.30), and the glue
// loads the .wasm of the same name; the other builds are never fetched.
const glue = [...new Set(fs.readFileSync(ortEntry, "utf8").match(/ort-wasm-simd-threaded[\w.]*\.mjs/g) ?? [])];
if (glue.length === 0) console.warn("[prepare-assets] found no wasm glue name in ort.webgpu.min.mjs");
fs.rmSync(path.join(app, "public/ort"), { recursive: true, force: true });
copy(ortEntry, path.join(app, "public/ort/ort.webgpu.min.mjs"), dropSourceMap);
for (const f of glue.flatMap((g) => [g, g.replace(/\.mjs$/, ".wasm")])) copy(path.join(ortDist, f), path.join(app, "public/ort", f), f.endsWith(".mjs") ? dropSourceMap : undefined);
for (const f of ["oversampler.mjs", "oversampler.wasm"]) copy(path.join(web, "packages/wasm-oversampler/prebuilt", f), path.join(app, "public/oversampler", f));

const kodak = path.join(web, "apps/demo/public/samples/kodak");
for (const f of fs.readdirSync(kodak).filter((f) => /\.(png|md)$/.test(f))) copy(path.join(kodak, f), path.join(app, "public/samples/kodak", f));
for (const f of fs.readdirSync(path.join(kodak, "thumbs"))) copy(path.join(kodak, "thumbs", f), path.join(app, "public/samples/kodak/thumbs", f));

const bundle = path.join(app, "public/models/web_bundle.json");
if (!fs.existsSync(bundle)) console.warn("[prepare-assets] public/models/web_bundle.json missing: run scripts/export_web_bundle.py --out-dir apps/demosr/public/models");
else if (!JSON.parse(fs.readFileSync(bundle, "utf8")).sr) console.warn("[prepare-assets] public/models/web_bundle.json is not a super-resolution bundle (no sr block)");
console.log("[prepare-assets] done");
