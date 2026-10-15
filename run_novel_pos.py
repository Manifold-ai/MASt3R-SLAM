"""Entry point for the novel-pos single-shot pipeline (ADR-001 修订2).

    python run_novel_pos.py --manifest manifest.yaml

Reads a manifest (existing GS; novel-view render image + GS-world render pointmap +
coverage + AI completion image; optional pose/intrinsics for sanity), degrades the GS to a
point cloud, fits a similarity transform C from the render frame (render world pointmap vs
MASt3R inference), applies C to the AI completion to place it in GS world, and writes a
merged `.ply`. A report-only validation gate measures render<->completion deviation.

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
    parser.add_argument("--strict-gate", action="store_true",
                        help="abort if the render<->completion validation gate exceeds tolerance")
    parser.add_argument("--debug-dir", default=None,
                        help="if set, dump the C-fit pointmaps (X_r_cam/X_r_GS), coverage/mask, "
                             "fit result + per-pixel residuals (fit_debug.npz) and two overlay PLYs")
    parser.add_argument("--new-only", action="store_true",
                        help="only append completion pixels that failed to match the render "
                             "(valid_match == False, i.e. new/fill content) instead of a "
                             "confidence percentile of the whole completion")
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

    run_pipeline(args.manifest, device=args.device, output_override=args.output,
                 strict_gate=args.strict_gate, debug_dir=args.debug_dir, new_only=args.new_only)


if __name__ == "__main__":
    main()
