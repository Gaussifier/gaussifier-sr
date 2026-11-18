/**
 * Before/after view on a 2D canvas: the left image up to a vertical divider, the right image
 * after it, both drawn into the same frame. Wheel or pinch zooms about the pointer, dragging
 * pans, dragging the handle (or anywhere on the divider line) moves the divider; `f` fits, `1`
 * shows image pixels at CSS-pixel size. Magnified pixels are drawn as squares (no smoothing),
 * so the result can be inspected pixel by pixel; minified images are smoothed.
 */

export interface Side { image: CanvasImageSource; width: number; height: number }

const HANDLE_RADIUS = 15; // CSS px
const GRAB = 10; // CSS px either side of the divider line

export class CompareView {
  private left: Side | null = null;
  private right: Side | null = null;
  /** Frame size in output pixels; both sides are scaled into it. */
  private frame = { width: 1, height: 1 };
  /** Canvas pixels per frame pixel, and the frame's top-left corner in canvas pixels. */
  private zoom = 1;
  private ox = 0;
  private oy = 0;
  /** Divider position as a fraction of the canvas width. */
  private split = 0.5;
  private fitted = true;
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private drag: "pan" | "split" | null = null;
  private pinch: { d: number; zoom: number } | null = null;
  private raf = 0;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly onSplit?: (fraction: number) => void) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas unavailable");
    this.ctx = ctx;
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.tabIndex = 0;
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    for (const t of ["pointerup", "pointercancel", "pointerleave"] as const) canvas.addEventListener(t, (e) => this.up(e));
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); const p = this.local(e); this.zoomAt(p.x, p.y, Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
    canvas.addEventListener("keydown", (e) => {
      if (e.key === "f") this.fit();
      else if (e.key === "1") this.actualSize();
      else if (e.key === "ArrowLeft" || e.key === "ArrowRight") { this.setSplit(this.split + (e.key === "ArrowLeft" ? -0.02 : 0.02)); e.preventDefault(); }
    });
    this.resize();
  }

  /** Show two images of the frame size; the view is kept when the frame size is unchanged. */
  set(left: Side, right: Side, width: number, height: number): void {
    const same = this.right && this.frame.width === width && this.frame.height === height;
    this.left = left; this.right = right; this.frame = { width, height };
    if (!same) this.fit(); else this.redraw();
  }

  setLeft(left: Side): void { this.left = left; this.redraw(); }

  fit(): void {
    const { width: cw, height: ch } = this.canvas;
    const m = 16 * devicePixelRatio;
    this.zoom = Math.max(1e-3, Math.min((cw - 2 * m) / this.frame.width, (ch - 2 * m) / this.frame.height));
    this.ox = (cw - this.frame.width * this.zoom) / 2;
    this.oy = (ch - this.frame.height * this.zoom) / 2;
    this.fitted = true;
    this.redraw();
  }

  actualSize(): void {
    const { width: cw, height: ch } = this.canvas;
    this.zoomAt(cw / 2, ch / 2, devicePixelRatio / this.zoom);
  }

  setSplit(fraction: number): void {
    this.split = Math.min(1, Math.max(0, fraction));
    this.onSplit?.(this.split);
    this.redraw();
  }

  get splitFraction(): number { return this.split; }

  private resize(): void {
    const r = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * devicePixelRatio)), h = Math.max(1, Math.round(r.height * devicePixelRatio));
    if (w === this.canvas.width && h === this.canvas.height) return;
    this.canvas.width = w; this.canvas.height = h;
    if (this.fitted) this.fit(); else this.redraw();
  }

  private zoomAt(x: number, y: number, factor: number): void {
    const fit = Math.min(this.canvas.width / this.frame.width, this.canvas.height / this.frame.height);
    const z = Math.min(64 * devicePixelRatio, Math.max(fit / 4, this.zoom * factor));
    this.ox = x - ((x - this.ox) * z) / this.zoom;
    this.oy = y - ((y - this.oy) * z) / this.zoom;
    this.zoom = z;
    this.fitted = false;
    this.redraw();
  }

  private local(e: { clientX: number; clientY: number }): { x: number; y: number } {
    const r = this.canvas.getBoundingClientRect();
    return { x: (e.clientX - r.left) * devicePixelRatio, y: (e.clientY - r.top) * devicePixelRatio };
  }

  private down(e: PointerEvent): void {
    this.canvas.setPointerCapture(e.pointerId);
    const p = this.local(e);
    this.pointers.set(e.pointerId, p);
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      this.pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), zoom: this.zoom };
      this.drag = null;
      return;
    }
    this.drag = Math.abs(p.x - this.split * this.canvas.width) <= GRAB * devicePixelRatio ? "split" : "pan";
    if (this.drag === "split") this.setSplit(p.x / this.canvas.width);
  }

  private move(e: PointerEvent): void {
    const p = this.local(e);
    const last = this.pointers.get(e.pointerId);
    if (!last) {
      this.canvas.style.cursor = Math.abs(p.x - this.split * this.canvas.width) <= GRAB * devicePixelRatio ? "ew-resize" : "grab";
      return;
    }
    this.pointers.set(e.pointerId, p);
    if (this.pinch && this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      this.zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, (this.pinch.zoom * d) / this.pinch.d / this.zoom);
    } else if (this.drag === "split") {
      this.setSplit(p.x / this.canvas.width);
    } else if (this.drag === "pan") {
      this.ox += p.x - last.x; this.oy += p.y - last.y;
      this.fitted = false;
      this.canvas.style.cursor = "grabbing";
      this.redraw();
    }
  }

  private up(e: PointerEvent): void {
    this.pointers.delete(e.pointerId);
    if (this.pointers.size < 2) this.pinch = null;
    if (this.pointers.size === 0) this.drag = null;
  }

  private redraw(): void {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => { this.raf = 0; this.draw(); });
  }

  private draw(): void {
    const { ctx, canvas } = this;
    const { width: cw, height: ch } = canvas;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (!this.left || !this.right) return;
    const x = Math.round(this.split * cw);
    const w = this.frame.width * this.zoom, h = this.frame.height * this.zoom;
    const side = (s: Side, x0: number, x1: number) => {
      if (x1 <= x0) return;
      ctx.save();
      ctx.beginPath(); ctx.rect(x0, 0, x1 - x0, ch); ctx.clip();
      // Square pixels when magnified, smoothing when minified.
      ctx.imageSmoothingEnabled = w / s.width < 1;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(s.image, this.ox, this.oy, w, h);
      ctx.restore();
    };
    side(this.left, 0, x);
    side(this.right, x, cw);
    // Divider: a light line with a dark halo, and a round handle with two arrows.
    const dpr = devicePixelRatio, cy = ch / 2, r = HANDLE_RADIUS * dpr;
    ctx.fillStyle = "rgba(14, 16, 19, 0.55)";
    ctx.fillRect(x - 2 * dpr, 0, 4 * dpr, ch);
    ctx.fillStyle = "#f2f4f7";
    ctx.fillRect(x - dpr, 0, 2 * dpr, ch);
    ctx.beginPath(); ctx.arc(x, cy, r, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(14, 16, 19, 0.85)"; ctx.fill();
    ctx.lineWidth = 2 * dpr; ctx.strokeStyle = "#f2f4f7"; ctx.stroke();
    ctx.fillStyle = "#f2f4f7";
    for (const dir of [-1, 1]) {
      ctx.beginPath();
      ctx.moveTo(x + dir * 9 * dpr, cy);
      ctx.lineTo(x + dir * 3.5 * dpr, cy - 5 * dpr);
      ctx.lineTo(x + dir * 3.5 * dpr, cy + 5 * dpr);
      ctx.closePath(); ctx.fill();
    }
  }
}
