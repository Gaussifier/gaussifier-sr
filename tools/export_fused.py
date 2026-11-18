"""Export a super-resolution checkpoint to the fused CUDA runtime's weight blobs.

    python tools/export_fused.py CHECKPOINT.pt --out DIR [--fm fm_fp16.gffm] [--self-check]

Writes into DIR (read by ``gaussifier_sr.fused.FusedSR.from_export``):

- ``head.gfch``: the trained correction head in FP16, every frozen GroupNorm folded into its
  conv. A back-projection head (``residual="backprojection"``: the head sees
  ``up(LR - down(render))``) is a GFCH v3 blob: its header records residual mode 1 and its
  first convs keep all 17 input channels in the training order (image 3, density, log_cov 3,
  rgb 3, rendered 3, residual 3, count). An image-residual head (``image - render``) is a GFCH
  v2 blob as in the base release: the residual is linear in the image and rendered channels,
  so it is folded into their weights (14 inputs). That fold is invalid for back-projection.
- ``fm.gffm``: the forward map (stage 1). With stage 1 frozen (no ``model`` state in the
  checkpoint) it is the base release's ``fm_fp16.gffm`` (checked against its SHA-256);
  otherwise it is exported from the checkpoint's ``model`` state (FP16).
- ``export.json``: residual mode, point-count rule and blob hashes.

Checkpoint: a dict with ``correction_head`` (the head's state dict: ``net.*`` main U-Net,
``aux_net.*`` aux U-Net, frozen GroupNorms as ``weight`` / ``bias`` / ``mean`` / ``var``),
optionally ``model`` (full state dict, when stage 1 was trained) and ``args`` (training
options: ``residual``, ``count_rule``, ``count_scale``, ``cascade_count``, ...). Without
``args["residual"]``, pass ``--residual``.

Base release files (``fm_fp16.gffm``; for ``--self-check`` also ``head.pt``, ``stage1.pt`` and
``head_fp16.gfch``) are read from ``$GAUSSIFIER_BASE_WEIGHTS`` or downloaded once into the
package cache. ``--self-check`` runs the release weights through the same code paths, which
must reproduce the release blobs byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gaussifier_sr.fused import RESIDUAL_MODES, export


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--residual",
        choices=sorted(RESIDUAL_MODES),
        default=None,
        help="override the checkpoint's args['residual']",
    )
    parser.add_argument(
        "--fm",
        type=Path,
        default=None,
        help="base release fm_fp16.gffm (default: base weights folder, downloaded once)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="check that the release weights reproduce the release blobs",
    )
    a = parser.parse_args()
    meta = export(a.checkpoint, a.out, residual=a.residual, fm=a.fm, check=a.self_check)
    json.dump(meta, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
