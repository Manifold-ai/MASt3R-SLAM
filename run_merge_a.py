"""Entry point for the ADR-002 A geometric merge (SLAM-C-003).

    python run_merge_a.py --manifest manifest_a.yaml

Seals one local cloud per keyframe (render + AI completion + known camera), solves the
Sim(3) of every keyframe with a MASt3R-SLAM style geometric graph anchored to P0 (the
degraded GS) and the known cameras -- no RGB residual -- and writes P0 plus every admitted
keyframe cloud, duplicates kept, to <output_dir>/merged.ply with transforms.json and
report.json.

Requires a CUDA GPU and the compiled `mast3r_slam_backends` extension.
"""
import argparse
import sys

import torch

from mast3r_slam.config import load_config, config


def main():
    parser = argparse.ArgumentParser(description="ADR-002 A: geometric T + duplicate-preserving merge")
    parser.add_argument("--manifest", required=True, help="path to the multi-keyframe manifest")
    parser.add_argument("--config", default="config/base.yaml",
                        help="MASt3R-SLAM config; local_opt thresholds/sigmas seed the solver")
    parser.add_argument("--output-dir", default=None, help="override manifest.output_dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vis-threshold", type=float, default=0.5,
                        help="tau in (0,1]: render pixels with alpha >= tau may anchor to P0 "
                             "(only when a keyframe provides visibility_map, C-002 format)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[merge_a] CUDA device not available. This pipeline requires a GPU and the "
                 "compiled `mast3r_slam_backends` extension (see setup.py). Aborting.")

    # the compiled matching/GN kernels launch on the CURRENT device (no device guard)
    dev = torch.device(args.device)
    torch.cuda.set_device(dev.index if dev.index is not None else torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    load_config(args.config)
    config["use_calib"] = False  # the known K enters through the solver, not the SLAM frames

    from mast3r_slam.novel_pos.pipeline_a import run_merge_a  # pulls in the CUDA backend

    run_merge_a(args.manifest, device=args.device, output_dir=args.output_dir,
                vis_threshold=args.vis_threshold)


if __name__ == "__main__":
    main()
