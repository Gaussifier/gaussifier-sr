# How the SR head was trained

The base model turns an image into 2D Gaussians in one forward pass. Stage 1 (a U-Net)
predicts a point-density map and per-pixel initial Gaussian shapes and colours. Points are
placed by equal-mass Voronoi sampling of the density. A small, weight-shared correction head
then refines the Gaussians over K = 7 rendered states, seeing the current render at each
state. For super-resolution, stage 1 is kept frozen and only the correction head is
fine-tuned, so that the Gaussians it produces from a low-resolution (LR) image render a sharp
high-resolution (HR) image.

## What the model sees

For an LR image `y` and scale `s = 2`:

| Input | Where it is used |
|---|---|
| `x = bicubic_up(y, s)` (clamped to [0, 1]) | stage 1 and the head's image channels |
| `y` itself | the head's residual channels, `up(y - down(render))` |

`down` is antialiased bicubic downsampling to the LR size and `up` is bicubic upsampling to
the HR size. The head input has the same 17 channels as the base model: image (3), density
(1), log-covariance map (3), colour map (3), current render (3), residual (3), and the hard
point count (1). The base model's residual is `x - render`. On a bicubic input this residual
asks the head to copy the blur. The back-projection residual is zero for every render that is
consistent with the LR observation. The head is therefore free to add detail that the LR
image does not rule out.

The number of Gaussians is `N = s^2 * own(y)`. Here `own(y)` is the model's own adaptive
count on the LR image (the integral of its predicted point rate). The HR grid therefore gets
the same number of Gaussians per LR pixel that the base model would spend on the LR image.
`N` is larger than the model's own count on the blurry input `x`. The initial covariances are
therefore shrunk by `own(x) / N`, which keeps the splatted mass constant.

## What it was trained against

The only target is the original HR crop. The loss is the L1 distance between the final render
(state 7) and the HR crop:

```
loss = mean |render_7(y) - HR|
```

No other term is used. There is no teacher network, no target Gaussian parameters, no
perceptual or adversarial loss, and no loss on the intermediate states. The LR image is
computed from the HR crop, so every HR image is a training pair and fresh pairs are made at
every step.

The renderer is differentiable. The gradient of the loss reaches the Gaussian positions,
covariances and colours of every state, and through the six head updates it reaches the head
weights (backpropagation through all seven states). Point placement and stage 1 are not
differentiated. Stage 1 is frozen, and the placement only fixes the initial points.

## Data

- HR images: DF2K, the 800 DIV2K training images and the 2,650 Flickr2K images.
- Each step draws 4 random 512 x 512 HR crops. Images are chosen in proportion to their area,
  and each crop is randomly flipped and transposed.
- The LR input is made on the GPU. It is the crop downsampled to 256 x 256 with antialiased
  bicubic and then quantized to 8 bits.

## Optimization

| Setting | Value |
|---|---|
| Trained parameters | correction head only (0.75 M), initialized from the base model |
| Frozen | stage 1 (backbone, density, shape and colour heads) |
| Optimizer | Adam, peak learning rate 4.5e-4, 300 warm-up steps, then constant |
| Batch | 4 crops of 512 x 512 |
| Gradient clipping | global norm 1.0 |
| Precision | head in bf16 autocast, everything else in fp32 |

The released head holds the weights after 6,000 steps, which is about 23 minutes of training
on one RTX 5090. These are the trained weights themselves, without weight averaging. Training
at this learning rate became unstable shortly after that step, so the planned cosine decay
was not reached.

## x4 and other scales

x4 has no separate model. It runs the x2 model twice. The second pass upsamples the first
pass's output, and its budget is `16 * own(y)` from the original LR image, so both passes keep
the LR image's Gaussian density.
