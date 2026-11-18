#!/usr/bin/env python
"""Export Gaussifier-SR for the browser engine as fp32 ONNX plus the WGSL head weights.

Writes into ``--out-dir``:

- ``forward_map.onnx``: stage 1 of the base release (v0.12.0), which the SR model leaves
  unchanged. The graph and the export settings are the base web bundle's.
- ``correction_head.onnx``, ``correction_head_wgsl.bin`` and ``correction_head_wgsl.json``: the
  SR correction head with every calibrated affine folded into its convolution. The head keeps
  its 17 input channels; channels 13 to 15 carry the back-projection residual
  ``up(LR - down(render))``, which the engine computes at every state.
- ``web_bundle.json``: the manifest, validated by ``schemas/web_bundle.schema.json``. Its
  ``sr`` block turns on the engine's ``upscale``.
- ``export_checks.json``: the ONNX graphs against the unfolded eager model on the CPU.

The base weights and ``sr_head.pt`` are downloaded on first use, as for ``gaussifier_sr.model``
(``GAUSSIFIER_BASE_WEIGHTS`` and ``--sr-head`` point to local copies). Needs ``onnx``,
``onnxruntime`` and ``jsonschema`` next to the package:

    uv run --with onnx --with onnxruntime --with jsonschema \\
        python web/scripts/export_web_bundle.py --out-dir web/apps/demosr/public/models
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gaussifier_sr.base.model import FirstStageModel, ForwardMapTuple, fold_frozen_gn
from gaussifier_sr.base.utils import PRODUCTION_DEFAULT_SEED, PRODUCTION_XY_STEP_PIXELS
from gaussifier_sr.base.voronoi_samplers import (
    DEFAULT_KNN_K,
    DEFAULT_MAX_MERGE_ROUNDS,
    FINAL_LLOYD_ITERS,
    OVERSAMPLE_FACTOR,
    PER_ROUND_REMOVE_FRACTION,
)
from gaussifier_sr.loop import POINTS_PER_PIXEL_MAX, SEED, STATES
from gaussifier_sr.model import base_weights_dir, load_model, sr_head_path

WEB_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = WEB_ROOT / "schemas/web_bundle.schema.json"
#: The gaussifier-sr release the bundle belongs to (the manifest's runtime_version).
RUNTIME_VERSION = "1.0.0"
#: The base release whose stage 1 and head architecture the SR model uses.
MODEL_VERSION = "0.12.0"
#: ``gaussifier_sr.loop.cascade``: pass k of a cascade runs with seed + 7919 k.
CASCADE_SEED_STRIDE = 7919
#: LR images are edge padded to this multiple: stage 1 pools three times.
SR_PAD_MULTIPLE = 8
FORWARD_MAP_OUTPUTS = ["density", "log_cov", "rgb", "raw_density", "rate"]
OPSET = 17
HEAD_WEIGHTS_BIN = "correction_head_wgsl.bin"
HEAD_WEIGHTS_LAYOUT = "correction_head_wgsl.json"
HEAD_TENSOR_COUNT = 44


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def assert_web_contract(model: FirstStageModel) -> None:
    config = model.config
    if config.decoder_density_aware:
        raise ValueError("web export requires decoder_density_aware=False")
    if model.correction_head is None or not bool(config.correction_predict_xy):
        raise ValueError("web export requires a correction head with predict_xy=True")
    if int(model.correction_head.output_channels) != 8:
        raise ValueError("web export requires an 8-channel correction head")
    if float(model.correction_xy_step_pixels) != PRODUCTION_XY_STEP_PIXELS:
        raise ValueError(
            f"web export requires correction_xy_step_pixels={PRODUCTION_XY_STEP_PIXELS}"
        )
    if int(config.correction_kernel_size) != 3 or int(config.correction_depth) != 2:
        raise ValueError("the WGSL head requires kernel_size=3 and depth=2")
    if int(config.correction_aux_hidden) <= 0:
        raise ValueError("the WGSL head requires an aux branch (correction_aux_hidden > 0)")


def head_weight_layout(model: FirstStageModel) -> tuple[dict, list[np.ndarray]]:
    """The folded head's weights in state_dict order, as the WGSL head reads them.

    Returns the layout (every ``offset`` and ``count`` in float32 elements) and the flattened
    arrays; the binary file is their concatenation.
    """
    head = model.correction_head
    config = model.config
    tensors: list[dict] = []
    arrays: list[np.ndarray] = []
    offset = 0
    for name, tensor in head.state_dict().items():
        array = tensor.detach().cpu().contiguous().to(torch.float32).numpy().reshape(-1)
        tensors.append(
            {
                "name": name,
                "shape": [int(dim) for dim in tensor.shape],
                "offset": offset,
                "count": int(array.size),
            }
        )
        arrays.append(array)
        offset += int(array.size)
    if len(tensors) != HEAD_TENSOR_COUNT:
        raise ValueError(f"expected {HEAD_TENSOR_COUNT} head tensors, got {len(tensors)}")
    layout = {
        "format": 2,
        "dtype": "float32",
        "arch": {
            "in_channels": int(head.INPUT_CHANNELS),
            "out_channels": int(head.output_channels),
            "hidden": int(config.correction_hidden),
            "aux_hidden": int(config.correction_aux_hidden),
            "depth": int(config.correction_depth),
            "kernel_size": int(config.correction_kernel_size),
        },
        "tensors": tensors,
    }
    return layout, arrays


def _file_entry(path: Path, role: str, shape: str) -> dict:
    """One ``files[]`` manifest entry: every artifact is f32 and identified by its sha256."""
    return {
        "path": path.name,
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
        "role": role,
        "precision": "f32",
        "shape": shape,
    }


def export_head_weights(model: FirstStageModel, out_dir: Path) -> list[dict]:
    """Write the WGSL head weight file and its layout; return their manifest entries."""
    layout, arrays = head_weight_layout(model)
    bin_path = out_dir / HEAD_WEIGHTS_BIN
    np.concatenate(arrays).astype("<f4").tofile(bin_path)
    layout_path = out_dir / HEAD_WEIGHTS_LAYOUT
    layout_path.write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    if bin_path.stat().st_size != 4 * sum(entry["count"] for entry in layout["tensors"]):
        raise RuntimeError("WGSL head weight file size does not match the layout")
    return [
        _file_entry(bin_path, "correction_head_wgsl", "any"),
        _file_entry(layout_path, "correction_head_wgsl_layout", "any"),
    ]


def _export(
    module: torch.nn.Module,
    example: torch.Tensor,
    path: Path,
    *,
    input_name: str,
    output_names: list[str],
    dynamic: bool,
) -> None:
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {name: {2: "H", 3: "W"} for name in [input_name, *output_names]}
    torch.onnx.export(
        module,
        (example,),
        str(path),
        dynamo=False,
        opset_version=OPSET,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def _verify(
    module: torch.nn.Module,
    path: Path,
    *,
    input_name: str,
    channels: int,
    sizes: tuple[tuple[int, int], ...],
    tolerance: float,
) -> dict[str, float]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    report: dict[str, float] = {}
    generator = torch.Generator().manual_seed(0)
    for height, width in sizes:
        example = torch.rand((1, channels, height, width), generator=generator)
        got = session.run(None, {input_name: example.numpy()})
        with torch.no_grad():
            expected = module(example)
        expected = list(expected) if isinstance(expected, (tuple, list)) else [expected]
        worst = max(float(np.abs(g - e.numpy()).max()) for g, e in zip(got, expected, strict=True))
        report[f"max_abs_diff_{height}x{width}"] = worst
        if worst > tolerance:
            raise RuntimeError(
                f"{path.name}: max abs diff {worst:.3e} at {height}x{width} exceeds {tolerance}"
            )
    return report


def export_bundle(
    out_dir: Path,
    *,
    sr_head: Path,
    static_sizes: tuple[int, ...] = (),
    trace_size: int = 256,
    verify_sizes: tuple[tuple[int, int], ...] = ((256, 256), (200, 136)),
    tolerance: float = 1e-3,
) -> dict:
    """Export both networks and write the validated manifest. Returns the manifest."""
    import jsonschema
    import onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = base_weights_dir()
    # The unfolded model is the reference the graphs are checked against.
    reference = load_model("cpu", sr_head=sr_head, fold=False, bf16=False)
    assert_web_contract(reference)
    model = fold_frozen_gn(reference)

    files: list[dict] = []
    checks: dict[str, dict[str, float]] = {}
    networks = [
        (
            "forward_map",
            ForwardMapTuple(model),
            ForwardMapTuple(reference),
            3,
            "image",
            FORWARD_MAP_OUTPUTS,
        ),
        (
            "correction_head",
            model.correction_head,
            reference.correction_head,
            17,
            "features",
            ["delta"],
        ),
    ]
    for role, module, eager, channels, input_name, outputs in networks:
        # One dynamic-shape graph plus a fixed graph per static size.
        variants = [("dynamic", None)] + [(f"static{size}", size) for size in static_sizes]
        for shape, size in variants:
            name = f"{role}.onnx" if size is None else f"{role}_static{size}.onnx"
            path = out_dir / name
            example_size = trace_size if size is None else size
            _export(
                module,
                torch.rand((1, channels, example_size, example_size)),
                path,
                input_name=input_name,
                output_names=outputs,
                dynamic=size is None,
            )
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            if any("Normalization" in node.op_type for node in graph.graph.node):
                raise RuntimeError(f"{path.name}: unexpected normalization op")
            checks[name] = _verify(
                eager,
                path,
                input_name=input_name,
                channels=channels,
                sizes=verify_sizes if size is None else ((size, size),),
                tolerance=tolerance,
            )
            files.append(_file_entry(path, role, shape))

    files.extend(export_head_weights(model, out_dir))

    manifest = {
        "schema_version": 2,
        "runtime_version": RUNTIME_VERSION,
        "model_version": MODEL_VERSION,
        "weights_sha256": {
            "stage1": sha256_of(weights_dir / "stage1.pt"),
            "head": sha256_of(sr_head),
        },
        "model": {
            "correction_iterations": STATES,
            "correction_predict_xy": True,
            "correction_output_channels": 8,
            "correction_input_channels": 17,
            "correction_xy_step_pixels": PRODUCTION_XY_STEP_PIXELS,
            "decoder_density_aware": False,
        },
        "files": files,
        "io": {
            "forward_map": {"input": "image", "outputs": FORWARD_MAP_OUTPUTS},
            "correction_head": {"input": "features", "output": "delta"},
        },
        "sampler": {
            "oversample": OVERSAMPLE_FACTOR,
            "merge_rounds": DEFAULT_MAX_MERGE_ROUNDS,
            "per_round_fraction": PER_ROUND_REMOVE_FRACTION,
            "knn_k": DEFAULT_KNN_K,
            "lloyd_iters": FINAL_LLOYD_ITERS,
            "tile_cap": 32,
            "seed_default": PRODUCTION_DEFAULT_SEED,
        },
        "renderer": {
            "tile": 16,
            "large_tile_threshold": 32,
            "alpha_cutoff": 1.0 / 255.0,
            "capacity": "max(8N,4096)",
        },
        "input": {"pad_multiple": 32, "pad_mode": "edge"},
        "sr": {
            "residual": "backprojection",
            "scale": 2,
            "count_rule": "s2",
            "cascade_count": "lr",
            "points_per_pixel_max": POINTS_PER_PIXEL_MAX,
            "seed": SEED,
            "cascade_seed_stride": CASCADE_SEED_STRIDE,
            "states": STATES,
            "pad_multiple": SR_PAD_MULTIPLE,
        },
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    for name, value in (("web_bundle.json", manifest), ("export_checks.json", checks)):
        (out_dir / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--sr-head", type=Path, default=None, help="sr_head.pt (default: the release, downloaded)"
    )
    parser.add_argument(
        "--static-sizes",
        type=int,
        nargs="*",
        default=[],
        help="also write fixed-size graphs for ORT graph capture, e.g. 512",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    sr_head = args.sr_head or sr_head_path()
    manifest = export_bundle(
        args.out_dir,
        sr_head=sr_head,
        static_sizes=tuple(args.static_sizes),
        tolerance=args.tolerance,
    )
    for entry in manifest["files"]:
        print(f"{entry['path']}: {entry['bytes']} bytes sha256 {entry['sha256'][:12]}")
    print(f"manifest: {args.out_dir / 'web_bundle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
