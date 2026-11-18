/**
 * MATLAB imresize(img, scale, 'bicubic') on 8-bit planar RGB, as gaussifier_sr.matlab_resize
 * implements it: Keys cubic (a = -0.5) contributions per output pixel, the kernel stretched by
 * 1 / scale when shrinking (antialiasing), weights normalised, symmetric edge padding, rows
 * first, in float64 with NumPy's summation order. It makes the demo's LR images from the Kodak
 * originals (the SR benchmarks' degradation) and the bicubic baseline the result is compared with.
 */

/** A planar [3, height, width] 8-bit RGB image. */
export interface Rgb8 { data: Uint8Array; width: number; height: number }

function cubic(x: number): number {
  const ax = Math.abs(x), ax2 = ax ** 2, ax3 = ax ** 3;
  return (1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1 ? 1 : 0) + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * (ax > 1 && ax <= 2 ? 1 : 0);
}

/** numpy's pairwise_sum for the short rows it is used on here (n <= 128). */
function npSum(a: Float64Array): number {
  const n = a.length;
  if (n < 8) { let s = 0; for (let i = 0; i < n; i++) s += a[i]; return s; }
  const r = a.slice(0, 8);
  let i = 8;
  for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[i + j];
  let s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
  for (; i < n; i++) s += a[i];
  return s;
}

/** Per output index: the input indices and normalised weights of its taps (_contributions). */
function contributions(inLength: number, outLength: number, scale: number): { taps: number; index: Int32Array; weight: Float64Array } {
  const width = scale < 1 ? 4 / scale : 4;
  const taps = Math.ceil(width) + 2;
  const index = new Int32Array(outLength * taps), weight = new Float64Array(outLength * taps);
  const row = new Float64Array(taps);
  for (let o = 0; o < outLength; o++) {
    const u = (o + 1) / scale + 0.5 * (1 - 1 / scale);
    const left = Math.floor(u - width / 2);
    for (let j = 0; j < taps; j++) {
      const d = u - (left + j);
      row[j] = scale < 1 ? scale * cubic(d * scale) : cubic(d);
      // Symmetric padding of the 1-based index: mirror over [0, 2 inLength).
      const m = (((left + j - 1) % (2 * inLength)) + 2 * inLength) % (2 * inLength);
      index[o * taps + j] = m < inLength ? m : 2 * inLength - 1 - m;
    }
    const total = npSum(row);
    for (let j = 0; j < taps; j++) weight[o * taps + j] = row[j] / total;
  }
  return { taps, index, weight };
}

/** Resize planar float64 [3, h, w] along one axis (0: rows, 1: columns). */
function resizeAxis(src: Float64Array, w: number, h: number, out: number, scale: number, axis: 0 | 1): Float64Array {
  const inLength = axis === 0 ? h : w;
  const { taps, index, weight } = contributions(inLength, out, scale);
  const W = axis === 0 ? w : out, H = axis === 0 ? out : h;
  const dst = new Float64Array(3 * W * H);
  for (let c = 0; c < 3; c++) {
    const s = c * w * h, d = c * W * H;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const o = axis === 0 ? y : x;
      let acc = 0;
      for (let j = 0; j < taps; j++) {
        const i = index[o * taps + j];
        acc += src[axis === 0 ? s + i * w + x : s + y * w + i] * weight[o * taps + j];
      }
      dst[d + y * W + x] = acc;
    }
  }
  return dst;
}

/** np.round: halves to even. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** imresize of an 8-bit image by `scale` (e.g. 1/2 or 4), rounded and clipped back to 8 bits. */
export function imresize8(image: Rgb8, scale: number): Rgb8 {
  const { width: w, height: h } = image;
  const outH = Math.ceil(h * scale), outW = Math.ceil(w * scale);
  const src = Float64Array.from(image.data, (v) => v / 255);
  const rows = resizeAxis(src, w, h, outH, scale, 0);
  const both = resizeAxis(rows, w, outH, outW, scale, 1);
  const data = Uint8Array.from(both, (v) => Math.min(255, Math.max(0, roundHalfEven(v * 255))));
  return { data, width: outW, height: outH };
}

/** gaussifier_sr.matlab_resize.matlab_downsample_uint8: the SR benchmarks' LR image. */
export function matlabDownsample(hr: Rgb8, scale: number): Rgb8 {
  return imresize8(hr, 1 / scale);
}
