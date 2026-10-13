"""Entry point for the novel-pos single-shot pipeline (ADR-001).

    python run_novel_pos.py --manifest manifest.yaml

Reads a manifest (existing GS, panorama + pose, novel-view render/completion images and
pose), degrades the GS to a point cloud, runs a pose-accuracy experiment on the AI
completion image (report only), updates the point cloud via MASt3R inference, and writes
a merged `.ply`.

Requires a CUDA GPU and the compiled `mast3r_slam_backends` extension: the pipeline
submodules transitively import it, so we fail fast here if CUDA is unavailable.
"""
import argparse
import sys

import torch

from mast3r_slam.config import load_config, config


def main():
    parser = argparse.ArgumentParser(
        description="Novel-view completion -> point-cloud update (ADR-001)"
    )
    parser.add_argument("--manifest", required=True, help="path to manifest.yaml")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--output", default=None, help="override manifest.output_ply")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit(
            "[novel_pos] CUDA device not available. This pipeline requires a GPU and the "
            "compiled `mast3r_slam_backends` extension (see setup.py). Aborting."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    load_config(args.config)
    config["use_calib"] = False  # v1 is strictly no-calib; intrinsics parsed but unused

    # Lazy import: these pull in mast3r_slam_backends (the CUDA extension).
    from mast3r_slam.novel_pos.pipeline import run_pipeline

    run_pipeline(args.manifest, device=args.device, output_override=args.output)


if __name__ == "__main__":
    main()
