#include <torch/extension.h>

#include "render.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &render_forward,
        "Render N 2D Gaussians to a float32 (C, H, W) image.",
        py::arg("means"),
        py::arg("scales"),
        py::arg("rotation"),
        py::arg("colors"),
        py::arg("height"),
        py::arg("width")
    );
}
