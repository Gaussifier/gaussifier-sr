# Web inference engine, super-resolution, and 2DGS viewer

Browser inference for the v0.12.0 frozen-GN model in FP32, the Gaussifier-SR
loop on top of it (`engine.upscale`, see [Super-resolution](#super-resolution)),
and a framework-free viewer for 2D Gaussian splat scenes. The super-resolution
bundle and goldens come from `scripts/` here. The base model's bundle and
goldens come from the Python side of the base repository, documented in its
[docs/web.md](https://github.com/Gaussifier/gaussifier/blob/main/docs/web.md).

## Layout

```
web/
  packages/wgsl/               shared WGSL shaders, GPU helper, placement kernels
  packages/wasm-oversampler/   Emscripten build of the CPU oversampler (bit-exact with CUDA); prebuilt/ is committed
  packages/inference/          engine: ONNX Runtime Web + WGSL placement, head, and K-loop
  packages/viewer/             viewer: loaders, exporters, WebGPU and WebGL2 backends, <gaussifier-viewer>
  apps/demo/                   the demo page (Vite): index.html, src/main.ts, src/style.css, src/inference.ts
  apps/demosr/                 the super-resolution demo (Vite): src/main.ts, src/compare.ts, src/resize.ts, src/inference.ts
  tools/                       server, Chromium launcher, browser test runner, demo checks, shared cli.mjs
  test/e2e.html                end-to-end gate against the goldens
  test/sr-e2e.html             super-resolution gate against the PyTorch goldens
  scripts/                     Python: the super-resolution bundle and goldens
  schemas/                     web_bundle.schema.json, the manifest (with the optional sr block)
  fixtures/                    goldens and reference inputs (large files ignored by git)
```

## Commands

```bash
cd web
npm install
npm run build                             # tsc for every package
npm run build:bundle                      # packages/viewer/bundle/gaussifier-viewer.js
npm run build:demo                        # apps/demo/dist, copies runtime assets into public/
npm test                                  # vitest unit tests (Node)
npm run test:browser                      # every WebGPU test page in Chromium, sequentially
npm run check:demo                        # waits for the built demo's automatic run (--scale S, --states K rerun with a change)
npm run serve                             # serves apps/demo/dist on :8080 and https :8443, all interfaces
npm run build:demosr                      # apps/demosr/dist, the super-resolution demo
npm run check:demosr                      # drives the built SR demo served at /demosr/ (--upload lr.png, --shot prefix)
npm run serve:demosr                      # serves apps/demosr/dist like `serve`
EMSDK=/path/to/emsdk npm run build:wasm   # only after editing the C++ oversampler source (needs Emscripten)
```

The oversampler wasm is committed under `packages/wasm-oversampler/prebuilt/`
with a `BUILD.json` that records the Emscripten version, the flags, and the
sha256 of every input. `npm test` compares those hashes with the current
sources, so editing `density_points.cpp`, `tools/transform.py`, or `build.sh`
without rebuilding fails the unit tests. The build is deterministic: two runs
of the same toolchain produce identical bytes. To rebuild, install the
recorded Emscripten once and `build.sh` finds it at `~/emsdk`:

```bash
git clone https://github.com/emscripten-core/emsdk ~/emsdk
~/emsdk/emsdk install 6.0.9 && ~/emsdk/emsdk activate 6.0.9
npm run build:wasm
```

The base model's bundle must exist at `apps/demo/public/models/` and its
goldens at `fixtures/<name>/` before the browser tests run; both come from the
base repository's scripts ([docs/web.md](https://github.com/Gaussifier/gaussifier/blob/main/docs/web.md)).
The super-resolution bundle goes to `apps/demosr/public/models/` and its goldens
to `fixtures/sr/` (see [Super-resolution](#super-resolution)).

`tools/browser-run.mjs <page>` runs one page; `--all` runs the list in the
script. Chromium is found through `CHROME_PATH` or the Playwright cache. On a
Linux workstation with an NVIDIA GPU the launcher starts Xvfb and passes
`--enable-features=Vulkan --use-vulkan=native`; plain headless Chromium on
Linux only reaches SwiftShader. `WEBGPU_SOFTWARE=1` selects SwiftShader.

## Serving to another machine

WebGPU needs a secure context. `npm run serve` opens an HTTPS port with a
self-signed certificate that names this host's addresses; accept it once in
the remote browser. Certificates are cached under `tools/.certs/`. Anyone who
can reach the port can download the model bundle.

## Viewer module

`gaussifier-viewer.js` is a single ES module with no dependencies. One tag
plays an asset from a URL; `apps/demo/public/embed.html` shows it in use.

```html
<script type="module" src="gaussifier-viewer.js"></script>
<gaussifier-viewer src="scene.splat2d"></gaussifier-viewer>
```

The element takes `src`, `backend`, and `state`, exposes `viewer`, `scene`,
and `load(url)`, and dispatches `load` and `error`. The same file exports the
programmatic API: `createViewer`, `loadSceneFromUrl`, `decodeScene`, the
loaders, and the exporters. Both backends draw the production simple-sum
formula, so a Splat2D export renders at 39.6 dB against its source.

Overlays (`viewer.setOverlay(name, on, opts)`): `image`, `diff` (with
`{ gain }`), `density`, `voronoi`, `centers`, `ellipses` (with `{ sigma,
width }`). The Voronoi overlay tints every cell from a hash of its owner and
draws dark one-pixel borders; the nearest-center map is computed once per
state on the CPU with an exact grid search (about 30 ms for 84K centers at
512²) and uploaded as an integer texture. The ellipse overlay draws each
Gaussian's 1σ ellipse as an antialiased ring of constant screen width in the
fragment shader, from the same instance buffer as the splats, with a dark
halo so it reads on any background. Centers are antialiased orange discs
with the same halo, drawn the same way. All three scale their detail with
zoom: borders fade out when cells are under about 8 display pixels, rings
fade to a hint under about 7 pixels of radius, and discs shrink to a
one-pixel stipple when centers are packed closer than a few pixels, so a
zoomed-out view shows the mosaic, the coverage, and the sampling density
rather than a wall of marks. `viewer.psnr()` scores a state
against the scene image; a state without stored render planes is rendered
by the CPU reference renderer on first use, so loaded assets can be scored
when the demo attaches the picked image.

Controls: wheel or pinch to zoom about the pointer, drag to pan (one finger
on touch screens; the canvas sets `touch-action: none`), `f` fits, `1` is
actual size, `[` and `]` step through states.

## Formats

- `.splat2d`: the Splat2D format, 12-byte header `GS2D`, count, height,
  width, then float32 xy, scale, rotation, and color. The stored scale is the
  reciprocal of the effective pixel sigma, clamped at 0.01. Default export.
  A padded scene is written at the original image size with xy rescaled so
  every Gaussian keeps its pixel position (`{ frame: "padded" }` keeps the
  padded frame instead).
- `.npz`: the `save_scene` layout of the Python package, read and written by
  the viewer; the Python workflow's format.
- The C++ harness's points, cov, and rgb NPY triple, read only.

## Engine

`createEngine({ bundleUrl })` fetches `web_bundle.json`, verifies every file
hash, creates the ONNX Runtime Web sessions on the WebGPU provider, and takes
the GPUDevice from the runtime. `engine.run(image, { states, count, seed })`
returns `RunResult`; with a super-resolution bundle, `engine.upscale(lr, {
scale })` returns `UpscaleResult`. Types live in `packages/inference/src/types.ts`; the
lower-level pieces are exported under `internals`.

The correction head has two backends behind one interface: `wgsl`, the
hand-written kernels fed by `correction_head_wgsl.bin`, and `ort`, the ONNX
session with graph capture at 512. `createEngine({ headBackend })` and
`run({ headBackend })` choose; `auto` prefers `wgsl`. The head is two UNets
of different widths with the same structure, and every kernel runs both
nets' matching layers in one dispatch from shared buffers with per-net
offsets. The folded 3×3 convolutions apply SiLU in the store, for 15 total
dispatches per head call. The conv tiling is a generator parameter.
Use `packages/inference/test/browser/head.html?sweep=1` to time the alternatives
and `?quick=1` to shorten the timing loop for SwiftShader.

The production profile is the only profile: raw density, truncated adaptive
count, capturable-native placement semantics, fixed initial Voronoi-mass
weights at every render, and the pixel-anchored xy step `2.56 / max(H, W)`.
Run options beyond that:

- `count` pins the Gaussian count instead of the adaptive `trunc(sum(rate))`,
  and `countScale` multiplies the adaptive count once the forward map has
  produced it; `RunResult.adaptiveCount` reports the model's own count either
  way. A count above the adaptive one scales the predicted covariance by
  `adaptiveCount / N` (the `cov_shift` of `pack_static`), as the Python and
  C++ paths do; without it N Gaussians of the predicted shape render about
  `N / adaptiveCount` times too bright. A lower count leaves the map alone.
- `keepStates` copies every state to a staging buffer inside the loop and
  reads them all back once at the end, so the GPU never stalls between
  states; `RunResult.snapshots` holds them. `onState` instead delivers each
  state as it completes, at one GPU round trip per state. `includeLatent:
  false` leaves the latent RGB out of either readback.
- A planar input marked `prepared` (the output of `prepareInput`) is used
  as is instead of being copied and clamped again.

The demo uses all three. There is no Run button: picking or dropping an
image runs it, and so does changing the Refine slider (correction steps,
0 to 11; the model is trained for 6, so K = refinements + 1 rendered
states) or the Count slider (a logarithmic scale from ×¼ to ×4 around the
model's own count, "auto" at the centre), with one more run queued if a
control changes during a run. A picked image larger than 768 px on its
longest side is downscaled to that on decode (the file card says so), so a
phone photo runs in the same time as a Kodak image. The picked image is prepared once, every refinement
is kept for the Refinement slider (each scored on first visit, Export saves
the shown one), and one of the 24 Kodak images (committed under
`apps/demo/public/samples/kodak`, 15 MB, with a thumbnail strip to pick
another; `?sample=kodim07` pins one) runs as soon as the page opens. The
engine is created and warmed at page open and the pill names the GPU: ORT
owns the device, and the engine asks for the high-performance adapter
because the default on a machine with two GPUs is often the integrated one.

## Super-resolution

A bundle whose `web_bundle.json` has an `sr` block carries the Gaussifier-SR
correction head, and `engine.upscale(lr, { scale })` runs
`gaussifier_sr.loop.upscale` on it: x2 is one pass, x4 two. A pass from y
(h x w) to 2h x 2w:

1. x = clamp(bicubic(y), 0, 1) on the GPU, PyTorch's `interpolate(scale_factor=2,
   mode="bicubic")` (A = -0.75, half-pixel centres, clamped source indices).
2. The forward map on x. Pass k places N = round(4^(k+1) own(LR)) Gaussians
   (Python's round, at most 0.4 per output pixel), where own(LR) is the sum of
   the rate of a forward map on the LR image itself, run once per call. Above
   the model's own count of x, trunc(sum(rate)), the log-covariance map shrinks
   by ln(N / own) as in the base engine's count rescale.
3. The base engine's placement (seed 1234, plus 7919 k in pass k), initial
   attributes, and K = 7 rendered states.
4. Before every head call, `sr/backprojection.ts` overwrites the features
   buffer's channels 13 to 15, where the splatter wrote image - render, with the
   back-projection residual up(y - down(render)). down is PyTorch's
   antialiased bicubic (`antialias=True`: Keys a = -0.5 stretched by the scale,
   windows truncated at the border and renormalised), up the plain bicubic.
   Both head backends read the 17 channels unfolded, so they need no other
   change. The fused CUDA runtime's `backprojection.cu` has the same kernels.
5. The output is the last render. LR images are edge padded to a multiple of
   8 (stage 1 pools three times) and the output is cropped back; LR sizes that
   are multiples of 8 run unpadded, exactly as in PyTorch.

Run from the repository root. The exporter needs `onnx`, `onnxruntime` and
`jsonschema` next to the package; the goldens need CUDA. The forward map it
writes is the base release's stage 1 with the base exporter's settings, so it
is byte-identical to the base bundle's `forward_map.onnx`. The SR head goes into
both `correction_head.onnx` and the WGSL weight file, its frozen GroupNorms
folded; no static-size graphs are written unless `--static-sizes 512` asks.

```bash
uv run --with onnx --with onnxruntime --with jsonschema python web/scripts/export_web_bundle.py \
  --out-dir web/apps/demosr/public/models
K=web/apps/demo/public/samples/kodak
uv run python web/scripts/export_web_goldens.py --out web/fixtures/sr --scale 2 4 \
  --image $K/kodim03.png --image $K/kodim05.png --image $K/kodim13.png \
  --image $K/kodim19.png --image $K/kodim23.png --image $K/kodim08.png:416x288
cd web && node tools/browser-run.mjs "test/sr-e2e.html?runs=5"
```

Each golden crops the HR image to a multiple of 8 x scale, makes the LR image
with the package's MATLAB bicubic port, and records the PyTorch reference
twice: as released (TF32 convolutions, the head under bf16 autocast) and in
IEEE FP32, plus the counts of every pass and the reference's run-to-run spread.
`test/sr-e2e.html` scores the browser against the HR image next to the
reference, the agreement with both references, the counts, and the median wall
time (`?head=ort` for the ONNX head, `?pinned=1` to also run with the
reference's counts). On an RTX 5090 in Chromium (Vulkan), five Kodak images at
x2 and x4 plus a 416 x 288 crop (LR sizes that are not multiples of 32) land
within 0.006 dB of the reference's PSNR, RGB and luma (0.002 dB mean absolute),
with the FP32 reference's counts exactly and the released reference's within 3
points. A x2 run to 768 x 512 takes about 62 ms and a x4 run about 88 ms
(PyTorch eager: 29 ms and 39 ms).

The demo in `apps/demosr` shows the result against the bicubic upscale (or the
original, or the input) with a draggable divider, zoom, and pan. The Kodak
samples are downscaled in the page with a port of the same MATLAB bicubic
(`src/resize.ts`, checked against Python by its unit test), so their PSNR is
the benchmark's; a picked file is the LR input itself. Every path is relative,
so the built site runs from any directory (`check:demosr` serves it at
`/demosr/`).

## Timing and parity

The head test checks the WGSL and ONNX heads on recorded CUDA inputs and
prints per-call timings at 512². The end-to-end tests report wall time,
PSNR against the input, and the delta from the CUDA golden score. Per-stage
timings in `RunResult.timings` measure submission unless
`run({ profile: true })` waits after each stage. The end-to-end tests allow
0.02% count variation (or at least two points) for differences in FP32
convolution and reduction.

## Conventions

- Packages emit ES modules to `dist/` with explicit `.js` import extensions so
  browser test pages load them without a bundler; test pages declare an
  import map for workspace packages and for `onnxruntime-web/webgpu`.
- Each package's `index.ts` is its API; byte-level and GPU-level pieces are
  exported under `internals` so tests and tools can reach them without the
  API growing.
- The UI vocabulary is the code's: "refinements" are the correction steps
  after the initializer (the engine's K rendered states are refinements + 1),
  and the demo's element ids and variables use the same words.
- Every compute kernel uses an explicit bind group layout and at most eight
  storage buffers per stage. Never bind one buffer twice in a dispatch.
- Record a whole stage into one command encoder and await once.
- ONNX Runtime Web runs its WASM single-threaded; the multithreaded build
  stalls on worker startup under test drivers and the WebGPU provider does
  not need it.
- fp16 is out of scope: the adapters used here expose no `shader-f16`.
