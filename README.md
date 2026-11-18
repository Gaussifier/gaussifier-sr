# Gaussifier-SR

Super-resolution inference code and weights for **Gaussifier: Fast 2D Gaussian
Decomposition**, an anonymous submission under double-blind review.

Gaussifier turns an image into 2D Gaussians in one forward pass. Here its correction head is
fine-tuned so that the Gaussians predicted from a low-resolution (LR) image render a sharp
high-resolution (HR) image. Only the 0.75 M-parameter correction head differs from the public
base model ([release v0.12.0](https://github.com/Gaussifier/gaussifier/releases/tag/v0.12.0)).
The head was trained with pairs made on the fly from HR crops, and its only loss is L1 to the
HR crop. [docs/supervision.md](docs/supervision.md) describes the inputs, the target and the
training recipe. Training code is not part of this release.

- Live browser demo (WebGPU): <https://gaussifier.github.io/demosr/>
- Base model and fused runtime: <https://github.com/Gaussifier/gaussifier>
- Project page: <https://gaussifier.github.io/>

## Results

PSNR / SSIM on the BT.601 luma, `scale` border pixels removed, MATLAB bicubic LR inputs
(`gaussifier_sr.evaluate`, PyTorch backend; the fused backend matches it to within 0.01 dB).
x4 runs the x2 model twice.

| Scale | Method | Set5 | Set14 | B100 | Urban100 |
|---|---|---|---|---|---|
| x2 | Bicubic | 33.68 / 0.9305 | 30.33 / 0.8700 | 29.56 / 0.8436 | 26.88 / 0.8407 |
| x2 | Gaussifier-SR | 37.26 / 0.9561 | 32.89 / 0.9095 | 31.68 / 0.8918 | 30.50 / 0.9085 |
| x4 | Bicubic | 28.43 / 0.8113 | 26.09 / 0.7050 | 25.96 / 0.6682 | 23.14 / 0.6582 |
| x4 | Gaussifier-SR | 31.11 / 0.8786 | 27.93 / 0.7656 | 27.14 / 0.7213 | 25.00 / 0.7459 |

Time per image on an RTX 5090 (median over 20 images, CUDA tensors in and out):

| Scale | Output size | Fused CUDA | PyTorch |
|---|---|---|---|
| x2 | 512 x 512 | 3.6 ms | 23.9 ms |
| x2 | 1024 x 768 | 11.2 ms | 65.8 ms |
| x2 | 2040 x 1356 | 39.8 ms | 238 ms |
| x4 | 2048 x 2048 | 72.6 ms | 461 ms |

## Installation

Requirements: Linux, an NVIDIA GPU, CUDA 13.x with `nvcc`, CMake 3.18 or newer (for the fused
runtime), Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Gaussifier/gaussifier-sr.git
cd gaussifier-sr
uv sync                    # add --extra metrics for LPIPS, DISTS and FID
```

The weights are downloaded on first use into `~/.cache/gaussifier_sr` (set
`GAUSSIFIER_SR_CACHE_DIR` to move it). The PyTorch backend uses the base model's `stage1.pt`
and `head.pt` from the base release and `sr_head.pt` from this repository's
[release](https://github.com/Gaussifier/gaussifier-sr/releases/tag/v1.0). The fused backend
uses the base release's `fm_fp16.gffm` and this release's `sr_head_fp16.gfch`. The renderer
and placement CUDA extensions are compiled on first use.

## Build

The PyTorch backend needs no build step. The fused CUDA runtime is built with CMake. Set
`CMAKE_CUDA_ARCHITECTURES` to your GPU (89 for Ada, 120 for Blackwell GeForce):

```bash
cmake -S cpp_inference/fused -B cpp_inference/build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build cpp_inference/build --target gaussifier_fused gaussifier_fused_infer --parallel
```

This builds `cpp_inference/build/libgaussifier_fused.so.1`, which `--backend fused` finds there
(set `GAUSSIFIER_SR_FUSED_LIBRARY` to use a library built elsewhere). The runtime is the base
model's fused runtime with one addition, the back-projection residual of the SR head. Pass
`-DCMAKE_CUDA_COMPILER=/path/to/nvcc` if CMake does not find CUDA. Details are in
[cpp_inference/fused/README.md](cpp_inference/fused/README.md).

## Inference

```bash
uv run python -m gaussifier_sr.infer --input lr.png --scale 2 --output sr.png
uv run python -m gaussifier_sr.infer --input lr_dir/ --scale 4 --output sr_dir/ --backend fused
```

x2 is one pass. x4 runs the x2 model twice, and the second pass keeps the LR image's Gaussian
density. `--backend torch` (the default) is the PyTorch reference implementation.
`--backend fused` is the CUDA runtime (see Build).

```python
import torch
from gaussifier_sr.loop import upscale
from gaussifier_sr.model import load_model

model = load_model("cuda")  # downloads the weights on first use
lr = torch.rand(3, 256, 256, device="cuda")  # RGB in [0, 1]
sr = upscale(model, lr, 2)  # (3, 512, 512)
```

## Evaluation

```bash
uv run python -m gaussifier_sr.evaluate --hr-dir benchmarks/Set5/HR --scale 2 4
uv run python -m gaussifier_sr.evaluate --hr-dir benchmarks/Urban100/HR --scale 2 4 --backend fused
```

Each HR image is cropped to a multiple of the scale and downsampled with a port of MATLAB
`imresize` (bicubic, antialiased). The port reproduces the MATLAB LR images of the standard
benchmarks exactly. `--backend bicubic` gives the MATLAB bicubic baseline, and `--lpips` and
`--dists` add perceptual metrics (`uv sync --extra metrics`).

## Browser demo

The browser engine in `web/` runs the same super-resolution loop with WebGPU and matches the
PyTorch reference to within 0.01 dB. Export the model for it, then build and serve the demo
(needs Node.js and npm):

```bash
uv run --with onnx --with onnxruntime --with jsonschema \
  python web/scripts/export_web_bundle.py --out-dir web/apps/demosr/public/models
cd web
npm ci && npm run build && npm run build:demosr
npm run serve:demosr
```

Open <http://localhost:8080/> in a browser with WebGPU (current Chrome or Edge). See
[web/README.md](web/README.md).
