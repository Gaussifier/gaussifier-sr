/**
 * Resampling kernels of the super-resolution loop, with PyTorch's CUDA semantics in FP32:
 *
 * - `bicubic_up`: F.interpolate(mode="bicubic", align_corners=False), Keys cubic with
 *   A = -0.75 and clamped source indices (upsample_bicubic2d_out_frame), optionally clamped to
 *   [0, 1]. It makes the pass input x = clamp(up(y)) and the head's residual channels.
 * - `aa_down_residual`: y - F.interpolate(render, mode="bicubic", antialias=True), the Keys
 *   cubic with a = -0.5 stretched by the scale, windows truncated at the border and their
 *   weights renormalised, horizontal sums first (upsample_gen2d_aa_out_frame).
 *
 * Between two states the head reads up(y - down(render)) in the channels where the base
 * engine's splatter wrote image - render; `recordResidual` overwrites them.
 */
import { Kernel, createStorageBuffer, createUniformBuffer } from "@gaussifier/wgsl";
import { FEATURES } from "../kloop/shaders.js";

const UP_WGSL = `
struct P { h: u32, w: u32, H: u32, W: u32, scale_h: f32, scale_w: f32, dst_off: u32, clamp01: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
const A = -0.75;
fn conv1(x: f32) -> f32 { return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0; }
fn conv2(x: f32) -> f32 { return ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A; }
// get_cubic_upsampling_coefficients
fn coeffs(t: f32) -> vec4<f32> { return vec4<f32>(conv2(t + 1.0), conv1(t), conv1(1.0 - t), conv2(1.0 - t + 1.0)); }
@compute @workgroup_size(16, 16) fn bicubic_up(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let y = g.y;
  if (x >= p.W || y >= p.H) { return; }
  // area_pixel_compute_source_index(scale, dst, align_corners=False, cubic=True)
  let real_x = p.scale_w * (f32(x) + 0.5) - 0.5;
  let real_y = p.scale_h * (f32(y) + 0.5) - 0.5;
  let in_x = i32(floor(real_x)); let in_y = i32(floor(real_y));
  let cx = coeffs(real_x - f32(in_x)); let cy = coeffs(real_y - f32(in_y));
  let xm = i32(p.w) - 1; let ym = i32(p.h) - 1;
  let x0 = u32(clamp(in_x - 1, 0, xm)); let x1 = u32(clamp(in_x, 0, xm));
  let x2 = u32(clamp(in_x + 1, 0, xm)); let x3 = u32(clamp(in_x + 2, 0, xm));
  let rows = vec4<u32>(u32(clamp(in_y - 1, 0, ym)), u32(clamp(in_y, 0, ym)), u32(clamp(in_y + 1, 0, ym)), u32(clamp(in_y + 2, 0, ym))) * p.w;
  for (var c = 0u; c < 3u; c++) {
    let b = c * p.h * p.w;
    var coef: vec4<f32>;
    for (var k = 0u; k < 4u; k++) {
      let r = b + rows[k];
      coef[k] = src[r + x0] * cx.x + src[r + x1] * cx.y + src[r + x2] * cx.z + src[r + x3] * cx.w;
    }
    var v = coef.x * cy.x + coef.y * cy.y + coef.z * cy.z + coef.w * cy.w;
    if (p.clamp01 != 0u) { v = clamp(v, 0.0, 1.0); }
    dst[p.dst_off + c * p.H * p.W + y * p.W + x] = v;
  }
}
`;

/** Taps of the antialiased window: 2 support + 1 with support = 2 scale, at scale 2. */
const AA_TAPS = 9;

const DOWN_WGSL = `
struct P { H: u32, W: u32, h: u32, w: u32, scale_h: f32, scale_w: f32, support_h: f32, support_w: f32, inv_h: f32, inv_w: f32, src_off: u32, pad0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> obs: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
const TAPS = ${AA_TAPS};
// upsample_antialias BicubicFilterFunctor (a = -0.5)
fn aa_cubic(t: f32) -> f32 {
  let a = -0.5; let x = abs(t);
  if (x < 1.0) { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0; }
  if (x < 2.0) { return (((x - 5.0) * x + 8.0) * x - 4.0) * a; }
  return 0.0;
}
struct Window { start: i32, size: i32, w: array<f32, TAPS> };
// _compute_weights_span and _compute_weights: input window of output index i, normalised weights.
fn aa_window(i: u32, in_size: u32, scale: f32, support: f32, inv: f32) -> Window {
  var win: Window;
  let center = scale * (f32(i) + 0.5);
  win.start = max(i32(center - support + 0.5), 0);
  win.size = min(i32(center + support + 0.5), i32(in_size)) - win.start;
  let start_m_center = f32(win.start) - center;
  var total = 0.0;
  for (var j = 0; j < TAPS; j++) {
    var v = 0.0;
    if (j < win.size) { v = aa_cubic((f32(j) + start_m_center + 0.5) * inv); }
    win.w[j] = v;
    total += v;
  }
  if (total != 0.0) { for (var j = 0; j < TAPS; j++) { win.w[j] = select(0.0, win.w[j] / total, j < win.size); } }
  return win;
}
@compute @workgroup_size(16, 16) fn aa_down_residual(@builtin(global_invocation_id) g: vec3<u32>) {
  let ox = g.x; let oy = g.y;
  if (ox >= p.w || oy >= p.h) { return; }
  let wx = aa_window(ox, p.W, p.scale_w, p.support_w, p.inv_w);
  let wy = aa_window(oy, p.H, p.scale_h, p.support_h, p.inv_h);
  let o = oy * p.w + ox;
  for (var c = 0u; c < 3u; c++) {
    let base = p.src_off + c * p.H * p.W + u32(wy.start) * p.W + u32(wx.start);
    var acc = 0.0;
    for (var y = 0; y < wy.size; y++) {
      let row = base + u32(y) * p.W;
      var hs = src[row] * wx.w[0];
      for (var j = 1; j < wx.size; j++) { hs += src[row + u32(j)] * wx.w[j]; }
      if (y == 0) { acc = hs * wy.w[0]; } else { acc += hs * wy.w[y]; }
    }
    out[c * p.h * p.w + o] = obs[c * p.h * p.w + o] - acc;
  }
}
`;

/**
 * The x2 pass's resampling: owns the low-resolution observation y and the residual scratch for
 * one pass size. Pipelines are compiled once per device (see `BackprojectionKernels`).
 */
export class BackprojectionKernels {
  readonly up: Kernel;
  readonly down: Kernel;
  constructor(readonly device: GPUDevice) {
    this.up = Kernel.create(device, { code: UP_WGSL, entryPoint: "bicubic_up", bindings: ["uniform", "read-only-storage", "storage"], label: "sr.bicubic_up" });
    this.down = Kernel.create(device, { code: DOWN_WGSL, entryPoint: "aa_down_residual", bindings: ["uniform", "read-only-storage", "read-only-storage", "storage"], label: "sr.aa_down_residual" });
  }
}

const groups = (n: number) => Math.ceil(n / 16);

/** Per-pass buffers: y (3 x h x w), the low-resolution residual, and the three parameter blocks. */
export class Backprojection {
  /** The pass's observation y, [3, h, w]; the caller uploads it. */
  readonly observation: GPUBuffer;
  private readonly residualLr: GPUBuffer;
  private readonly upInput: GPUBuffer;
  private readonly upResidual: GPUBuffer;
  private readonly down: GPUBuffer;

  /** One x2 pass from h x w to H x W (PyTorch's float scales: h / H up, H / h down). */
  constructor(private readonly k: BackprojectionKernels, readonly w: number, readonly h: number, readonly W: number, readonly H: number) {
    const d = k.device;
    this.observation = createStorageBuffer(d, 3 * w * h * 4, "sr.observation");
    this.residualLr = createStorageBuffer(d, 3 * w * h * 4, "sr.residual_lr");
    const up = (dstOff: number, clamp01: number) => {
      const b = new ArrayBuffer(32), u = new Uint32Array(b), f = new Float32Array(b);
      u[0] = h; u[1] = w; u[2] = H; u[3] = W; f[4] = h / H; f[5] = w / W; u[6] = dstOff; u[7] = clamp01;
      return createUniformBuffer(d, new Uint8Array(b), "sr.up");
    };
    this.upInput = up(0, 1);
    this.upResidual = up(FEATURES.diff * H * W, 0);
    // Host-side scales as PyTorch derives them for size=: FP32 input / output; support 2 scale.
    const b = new ArrayBuffer(48), u = new Uint32Array(b), f = new Float32Array(b);
    u[0] = H; u[1] = W; u[2] = h; u[3] = w;
    f[4] = H / h; f[5] = W / w; f[6] = 2 * f[4]; f[7] = 2 * f[5]; f[8] = 1 / f[4]; f[9] = 1 / f[5];
    u[10] = FEATURES.rendered * H * W;
    if (f[4] > 2 || f[5] > 2) throw new RangeError("the antialiased window supports scales up to 2");
    this.down = createUniformBuffer(d, new Uint8Array(b), "sr.down");
  }

  /** x = clamp(bicubic(y), 0, 1) into `image` ([3, H, W]), the pass input. */
  recordInput(encoder: GPUCommandEncoder, image: GPUBuffer): void {
    const pass = encoder.beginComputePass({ label: "sr-input" });
    this.k.up.dispatch(pass, [this.upInput, this.observation, image], groups(this.W), groups(this.H));
    pass.end();
  }

  /** Replace image - render in the features buffer with up(y - down(render)). */
  recordResidual(encoder: GPUCommandEncoder, features: GPUBuffer): void {
    const pass = encoder.beginComputePass({ label: "sr-residual" });
    this.k.down.dispatch(pass, [this.down, features, this.observation, this.residualLr], groups(this.w), groups(this.h));
    this.k.up.dispatch(pass, [this.upResidual, this.residualLr, features], groups(this.W), groups(this.H));
    pass.end();
  }

  dispose(): void {
    for (const b of [this.observation, this.residualLr, this.upInput, this.upResidual, this.down]) b.destroy();
    this.k.up.clearCache();
    this.k.down.clearCache();
  }
}
