import { createEngine, type Engine, type PlanarImage, type UpscaleResult } from "@gaussifier/inference";
import { createOversampler } from "@gaussifier/wasm-oversampler";
import type { Rgb8 } from "./resize.js";

const base = (rel: string) => new URL(rel, document.baseURI).href;
let enginePromise: Promise<Engine> | null = null;

/**
 * The engine is created once per page; the bundle and runtime files live next to index.html.
 * The warmup runs x4 (which visits every forward-map size of x2 as well) on a blank image of
 * each Kodak orientation, so the first sample does not pay for shader compilation.
 */
export function getEngine(): Promise<Engine> {
  enginePromise ??= (async () => {
    const engine = await createEngine({
      bundleUrl: base("models/"),
      ortWasmPaths: base("ort/"),
      oversampler: () => createOversampler({ moduleUrl: base("oversampler/oversampler.mjs"), wasmUrl: base("oversampler/oversampler.wasm") }),
    });
    for (const [w, h] of [[192, 128], [128, 192]]) {
      await engine.upscale({ data: new Float32Array(3 * w * h).fill(0.5), width: w, height: h }, { scale: 4, states: 2, counts: [64, 64] });
    }
    return engine;
  })();
  return enginePromise;
}

/**
 * What the adapter ORT chose means for run time (as in the regular demo): "fast" is a discrete
 * or Apple GPU, "slow" an integrated Intel GPU, "software" a CPU renderer such as SwiftShader.
 */
export type GpuSpeed = "fast" | "slow" | "software" | "unknown";
export function gpuSpeed(adapter: string): GpuSpeed {
  const a = adapter.toLowerCase();
  if (/swiftshader|llvmpipe|lavapipe|software|\bwarp\b|cpu/.test(a)) return "software";
  if (/nvidia|amd|radeon|apple|qualcomm|adreno|\barm\b|mali/.test(a)) return "fast";
  if (/intel|integrated|iris|uhd|gen-?\d/.test(a)) return "slow";
  return "unknown";
}

/**
 * The longest side of an output. A picked image is the LR input; when it would come out larger
 * it is downscaled on decode, so a phone photo runs in about the time of a Kodak image and fits
 * the memory of an integrated GPU.
 */
export const MAX_OUTPUT_SIDE = 1024;
/** Smallest LR side the model is run on. */
export const MIN_INPUT_SIDE = 16;

/** The size a picked image is used at for `scale`: unchanged when the output fits, else scaled to fit. */
export function inputSize(width: number, height: number, scale: number): { width: number; height: number; resized: boolean } {
  const longest = Math.max(width, height), limit = Math.floor(MAX_OUTPUT_SIDE / scale);
  if (longest <= limit) return { width, height, resized: false };
  const s = limit / longest;
  return { width: Math.max(1, Math.round(width * s)), height: Math.max(1, Math.round(height * s)), resized: true };
}

function toRgb8(bitmap: ImageBitmap): Rgb8 {
  const { width, height } = bitmap;
  const ctx = new OffscreenCanvas(width, height).getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, width, height);
  const hw = width * height, out = new Uint8Array(3 * hw);
  for (let i = 0; i < hw; i++) { out[i] = data[4 * i]; out[hw + i] = data[4 * i + 1]; out[2 * hw + i] = data[4 * i + 2]; }
  return { data: out, width, height };
}

/** A sample PNG as stored, without colour management, so its pixels are the benchmark's. */
export async function decodeSample(url: string): Promise<Rgb8> {
  const blob = await (await fetch(url)).blob();
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  try { return toRgb8(bitmap); } finally { bitmap.close(); }
}

/** A picked file at its input size for `scale` (see `inputSize`). */
export async function decodeUpload(file: File, scale: number): Promise<Rgb8> {
  const probe = await createImageBitmap(file);
  const size = inputSize(probe.width, probe.height, scale);
  let bitmap = probe;
  if (size.resized) {
    bitmap = await createImageBitmap(probe, { resizeWidth: size.width, resizeHeight: size.height, resizeQuality: "high" });
    probe.close();
  }
  try { return toRgb8(bitmap); } finally { bitmap.close(); }
}

export function toPlanar(image: Rgb8): PlanarImage {
  return { data: Float32Array.from(image.data, (v) => v / 255), width: image.width, height: image.height };
}

/** The 8-bit image a result is shown and saved as: round(clamp(x) * 255), halves to even. */
export function quantize(result: UpscaleResult): Rgb8 {
  const data = new Uint8Array(result.image.length);
  for (let i = 0; i < data.length; i++) {
    const v = Math.fround(Math.min(1, Math.max(0, result.image[i])) * 255), r = Math.round(v);
    data[i] = r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }
  return { data, width: result.width, height: result.height };
}

/** PSNR in dB over the three channels of two 8-bit images of the same size. */
export function psnr(a: Rgb8, b: Rgb8): number {
  let se = 0;
  for (let i = 0; i < a.data.length; i++) { const d = a.data[i] - b.data[i]; se += d * d; }
  return se === 0 ? Infinity : 10 * Math.log10((255 * 255 * a.data.length) / se);
}

/** An 8-bit image on a canvas, for drawing and saving. */
export function toCanvas(image: Rgb8): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = image.width; canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2d context unavailable");
  const pixels = ctx.createImageData(image.width, image.height);
  const hw = image.width * image.height;
  for (let i = 0; i < hw; i++) {
    pixels.data[4 * i] = image.data[i]; pixels.data[4 * i + 1] = image.data[hw + i]; pixels.data[4 * i + 2] = image.data[2 * hw + i]; pixels.data[4 * i + 3] = 255;
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas;
}
