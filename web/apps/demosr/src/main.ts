import "./style.css";
import { WebGpuUnavailableError, type UpscaleResult } from "@gaussifier/inference";
import { CompareView, type Side } from "./compare.js";
import { MIN_INPUT_SIDE, decodeSample, decodeUpload, getEngine, gpuSpeed, inputSize, psnr, quantize, toCanvas, toPlanar, type GpuSpeed } from "./inference.js";
import { imresize8, matlabDownsample, type Rgb8 } from "./resize.js";

// ---- elements ----
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const ui = {
  canvas: $<HTMLCanvasElement>("view"),
  stage: $<HTMLElement>("stage"),
  empty: $<HTMLDivElement>("empty"),
  live: $<HTMLDivElement>("live"),
  labelLeft: $<HTMLDivElement>("labelLeft"),
  labelRight: $<HTMLDivElement>("labelRight"),
  gpuPill: $<HTMLSpanElement>("gpuPill"),
  runStatus: $<HTMLDivElement>("runStatus"),
  status: $<HTMLDivElement>("status"),
  imageDrop: $<HTMLDivElement>("imageDrop"),
  imageInput: $<HTMLInputElement>("image"),
  thumb: $<HTMLImageElement>("thumb"),
  imageName: $<HTMLDivElement>("imageName"),
  imageDims: $<HTMLDivElement>("imageDims"),
  samples: $<HTMLDivElement>("samples"),
  scale: $<HTMLDivElement>("scale"),
  compare: $<HTMLDivElement>("compare"),
  psnr: $<HTMLSpanElement>("psnr"),
  psnrBicubic: $<HTMLSpanElement>("psnrBicubic"),
  statN: $<HTMLDivElement>("statN"),
  time: $<HTMLDivElement>("time"),
  save: $<HTMLButtonElement>("save"),
};

// ---- state ----
/** The picked image: a Kodak sample (the original is known and the LR input is made from it) or a file (the LR input itself). */
interface Source { name: string; original: Rgb8 | null; file: File | null }
type Compare = "bicubic" | "original" | "input";
interface Shown { name: string; scale: number; lr: Rgb8; result: UpscaleResult; sides: Record<Compare, Side | null> & { sr: Side } }
let source: Source | null = null;
let scale = 2;
let compare: Compare = "bicubic";
let shown: Shown | null = null;
let running = false;
let queued = false;

const view = new CompareView(ui.canvas, (f) => syncLabels(f));
const fmtInt = (n: number) => n.toLocaleString("en-US");
const fmtDb = (v: number | null) => (v === null ? "—" : `${Number.isFinite(v) ? v.toFixed(2) : "∞"}<small>dB</small>`);

// ---- engine and GPU status ----
// The engine loads as soon as the page opens (14 MB of model files and the runtime, shader
// compilation, a warmup run per sample orientation), so the first run does not pay for it.
let engineReady = false, engineError: string | null = null;
let speed: GpuSpeed = "unknown";
const SPEED_NOTE: Record<GpuSpeed, string> = {
  fast: "",
  slow: "Runs take a few seconds on an integrated GPU.",
  software: "No GPU: this is a software renderer, runs take minutes.",
  unknown: "",
};
function setPill(text: string, css: "on" | "warn" | "bad", title: string): void {
  ui.gpuPill.textContent = text;
  ui.gpuPill.classList.remove("on", "warn", "bad");
  ui.gpuPill.classList.add(css);
  ui.gpuPill.title = title;
}
const NEEDS_WEBGPU = "This page needs WebGPU: a current Chrome or Edge, Safari 26, or Firefox on Windows, with a GPU.";
function updateReady(): void {
  if (running) return;
  if (engineError) { ui.runStatus.textContent = engineError; return; }
  if (!engineReady) { ui.runStatus.textContent = source ? "Loading the model…" : "Loading the model and the sample image…"; return; }
  const note = SPEED_NOTE[speed];
  ui.runStatus.textContent = (source ? "Ready." : "Pick an image to start.") + (note ? ` ${note}` : "");
}
getEngine().then((engine) => {
  engineReady = true;
  speed = gpuSpeed(engine.adapter);
  const vendor = engine.adapter.split(" · ").pop() ?? "";
  const detail = `${engine.adapter}. A x2 run of a Kodak image takes well under a second on a discrete GPU, a few seconds on an integrated one, and minutes on a software renderer.`;
  if (speed === "fast") setPill(`${vendor} gpu`, "on", detail);
  else if (speed === "slow") setPill("integrated gpu", "warn", detail);
  else if (speed === "software") setPill("no gpu · slow", "bad", detail);
  else setPill("gpu", "on", detail);
  updateReady();
}).catch((e) => {
  engineError = e instanceof WebGpuUnavailableError ? NEEDS_WEBGPU : `${NEEDS_WEBGPU} The engine failed to load: ${String(e)}`;
  setPill("no webgpu", "bad", NEEDS_WEBGPU);
  updateReady();
});

function showStatus(text: string): void { ui.status.hidden = false; ui.status.textContent = text; }

// ---- running ----
/** The LR input for a scale: the original downscaled with MATLAB bicubic, or the picked file. */
async function inputFor(src: Source, s: number): Promise<Rgb8> {
  if (src.original) return matlabDownsample(src.original, s);
  return decodeUpload(src.file!, s);
}

// There is no button: picking an image or changing the scale runs, and a change made during a
// run queues one more run with the latest settings.
async function run(): Promise<void> {
  if (!source) return;
  if (engineError) { updateReady(); return; }
  if (running) { queued = true; return; }
  running = true;
  const src = source, s = scale;
  ui.runStatus.textContent = engineReady ? "Running…" : "Loading the model, then running…";
  ui.live.hidden = false; ui.live.textContent = s === 2 ? "running ×2" : "running ×2, ×2";
  try {
    const lr = await inputFor(src, s);
    if (Math.min(lr.width, lr.height) < MIN_INPUT_SIDE) throw new Error(`the image is smaller than ${MIN_INPUT_SIDE} pixels on a side`);
    const engine = await getEngine();
    const result = await engine.upscale(toPlanar(lr), { scale: s });
    if (src !== source || s !== scale) return; // a newer pick or scale takes over (it queued a run)
    show(src, s, lr, result);
    ui.runStatus.textContent = `Done in ${fmtMs(result.timings.total)} on ${engine.adapter}.`;
  } catch (e) {
    ui.runStatus.textContent = engineError ?? `Run failed: ${String(e)}`;
  } finally {
    running = false; ui.live.hidden = true;
    ui.runStatus.dataset.runs = String(Number(ui.runStatus.dataset.runs ?? 0) + 1);
    if (queued) { queued = false; void run(); }
  }
}
const fmtMs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

function show(src: Source, s: number, lr: Rgb8, result: UpscaleResult): void {
  const sr = quantize(result);
  const bicubic = imresize8(lr, s);
  const side = (image: Rgb8): Side => ({ image: toCanvas(image), width: image.width, height: image.height });
  // A known original is the HR image the LR input was made from: same frame as the output.
  const original = src.original && src.original.width === sr.width && src.original.height === sr.height ? src.original : null;
  shown = { name: src.name, scale: s, lr, result, sides: { sr: side(sr), bicubic: side(bicubic), original: original ? side(original) : null, input: side(lr) } };
  ui.psnr.innerHTML = fmtDb(original ? psnr(sr, original) : null);
  ui.psnrBicubic.innerHTML = fmtDb(original ? psnr(bicubic, original) : null);
  for (const el of [ui.psnr, ui.psnrBicubic]) el.parentElement!.title = original ? "" : "No original for a picked image: PSNR needs the high-resolution image.";
  const passes = result.passes;
  ui.statN.textContent = fmtInt(passes[passes.length - 1].n);
  ui.statN.title = passes.map((p) => `${p.width}×${p.height}: ${fmtInt(p.n)} Gaussians`).join("\n");
  ui.time.textContent = fmtMs(result.timings.total);
  ui.time.title = `Wall time of the upscale call (${passes.length} ×2 pass${passes.length > 1 ? "es" : ""} of 7 states, read back to the page); decoding, the bicubic baseline and PSNR are not included.`;
  if (!shown.sides[compare]) compare = "bicubic";
  syncCompare();
  view.set(shown.sides[compare]!, shown.sides.sr, sr.width, sr.height);
  ui.empty.hidden = true;
  ui.status.hidden = true;
  ui.labelLeft.hidden = ui.labelRight.hidden = false;
  syncLabels(view.splitFraction);
}

// ---- controls: scale and comparison ----
function pressed(group: HTMLElement, attr: string, value: string): void {
  for (const b of group.querySelectorAll<HTMLButtonElement>("button")) b.setAttribute("aria-pressed", String(b.dataset[attr] === value));
}
for (const b of ui.scale.querySelectorAll<HTMLButtonElement>("button")) {
  b.addEventListener("click", () => {
    const s = Number(b.dataset.scale);
    if (s === scale) return;
    scale = s; pressed(ui.scale, "scale", String(s)); updateCard(); void run();
  });
}
const LABELS: Record<Compare, string> = { bicubic: "Bicubic", original: "Original", input: "Input" };
function syncCompare(): void {
  pressed(ui.compare, "compare", compare);
  const original = ui.compare.querySelector<HTMLButtonElement>('button[data-compare="original"]')!;
  original.disabled = !!shown && !shown.sides.original;
  original.title = original.disabled ? "Only the Kodak samples come with their original." : "The high-resolution image the input was made from.";
  ui.labelLeft.textContent = LABELS[compare];
}
for (const b of ui.compare.querySelectorAll<HTMLButtonElement>("button")) {
  b.addEventListener("click", () => {
    const c = b.dataset.compare as Compare;
    if (!shown || !shown.sides[c]) return;
    compare = c; syncCompare();
    view.setLeft(shown.sides[c]!);
  });
}
function syncLabels(fraction: number): void {
  ui.labelLeft.classList.toggle("faded", fraction < 0.12);
  ui.labelRight.classList.toggle("faded", fraction > 0.88);
}

// ---- controls: the source image ----
function updateCard(): void {
  if (!source) return;
  if (source.original) {
    const o = source.original;
    ui.imageDims.textContent = `${o.width}×${o.height} · input ${o.width / scale}×${o.height / scale}, MATLAB bicubic ÷${scale}`;
  } else if (source.file) {
    const w = ui.thumb.naturalWidth, h = ui.thumb.naturalHeight;
    if (!w || !h) return;
    const size = inputSize(w, h, scale);
    const resized = size.resized ? ` · resized to ${size.width}×${size.height}` : "";
    ui.imageDims.textContent = `${w}×${h} · ${(source.file.size / 1024).toFixed(0)} KB${resized}`;
  }
}
function setSource(src: Source, thumbUrl: string): void {
  source = src;
  ui.thumb.onload = updateCard;
  ui.thumb.src = thumbUrl;
  ui.imageName.textContent = src.name;
  updateCard();
  updateReady();
  void run();
}
function setFile(file: File): void {
  for (const b of ui.samples.querySelectorAll("button")) b.setAttribute("aria-pressed", "false");
  setSource({ name: file.name, original: null, file }, URL.createObjectURL(file));
}
ui.imageDrop.addEventListener("click", () => ui.imageInput.click());
ui.imageInput.addEventListener("change", () => { const f = ui.imageInput.files?.[0]; if (f) setFile(f); });
// Drop an image anywhere on the stage to run it.
ui.stage.addEventListener("dragover", (e) => { e.preventDefault(); ui.stage.classList.add("over"); });
ui.stage.addEventListener("dragleave", () => ui.stage.classList.remove("over"));
ui.stage.addEventListener("drop", (e) => {
  e.preventDefault(); ui.stage.classList.remove("over");
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) setFile(f); else if (f) showStatus("Drop an image file.");
});

// ---- controls: Kodak samples, a strip of 24 thumbnails; one at random on page open, or ?sample=kodim07 ----
const KODAK = Array.from({ length: 24 }, (_, i) => `kodim${String(i + 1).padStart(2, "0")}`);
for (const name of KODAK) {
  const b = document.createElement("button");
  b.type = "button"; b.dataset.sample = name; b.title = name; b.setAttribute("role", "option"); b.setAttribute("aria-pressed", "false");
  const img = document.createElement("img"); img.src = `samples/kodak/thumbs/${name}.jpg`; img.alt = name; img.loading = "lazy";
  b.append(img);
  b.addEventListener("click", () => void loadSample(name));
  ui.samples.append(b);
}
async function loadSample(name: string): Promise<void> {
  for (const b of ui.samples.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.sample === name));
  try {
    const url = new URL(`samples/kodak/${name}.png`, document.baseURI).href;
    const original = await decodeSample(url);
    setSource({ name: `${name}.png`, original, file: null }, url);
    ui.samples.querySelector(`button[data-sample="${name}"]`)?.scrollIntoView({ block: "nearest", inline: "center" });
  } catch (e) {
    showStatus(`Could not load ${name}: ${String(e)}`); updateReady();
  }
}
const query = new URLSearchParams(location.search);
if (query.get("scale") === "4") { scale = 4; pressed(ui.scale, "scale", "4"); }
const requested = query.get("sample");
void loadSample(requested && KODAK.includes(requested) ? requested : KODAK[Math.floor(Math.random() * KODAK.length)]);

// ---- save ----
ui.save.addEventListener("click", () => {
  if (!shown) { showStatus("Nothing to save yet."); return; }
  const canvas = shown.sides.sr.image as HTMLCanvasElement;
  const name = `${shown.name.replace(/\.[^.]+$/, "")}_x${shown.scale}_gaussifier-sr.png`;
  canvas.toBlob((blob) => {
    if (!blob) { showStatus("Could not encode the PNG."); return; }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    showStatus(`Saved ${name}, ${canvas.width}×${canvas.height}.`);
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, "image/png");
});
