"""Single-shot orchestration for the novel-pos pipeline (ADR-001).

Flow: load manifest -> degrade GS to points -> load model -> build render/completion
frames at the render pose -> pose-accuracy experiment (report only) -> point-cloud
update -> write merged PLY.

WARNING: imports the CUDA backend (via the submodules below). GPU-only.
"""
from mast3r_slam.mast3r_utils import load_mast3r

from mast3r_slam.novel_pos.manifest import load_manifest
from mast3r_slam.novel_pos.gs_io import (
    read_gaussian_ply,
    degrade_gaussians_to_points,
    write_ply,
)
from mast3r_slam.novel_pos.pose_utils import sim3_from_pose_spec
from mast3r_slam.novel_pos.frames import load_rgb01, build_frame
from mast3r_slam.novel_pos.pose_experiment import run_pose_accuracy_experiment
from mast3r_slam.novel_pos.pc_update import update_point_cloud


def _print_pose_report(dev: dict) -> None:
    print("=" * 60)
    print("Pose Accuracy Experiment (report only -- does NOT gate the update)")
    if not dev.get("ok"):
        print(f"  FAILED: {dev.get('reason')}  (match_frac={dev.get('match_frac')})")
    else:
        print(f"  matches used          : {dev['match_frac'] * 100:.1f}% of pixels")
        print(f"  rotation deviation    : {dev['rotation_deg']:.3f} deg")
        print(f"  translation deviation : {dev['translation_norm']:.4f} (world units)")
        print(f"  scale deviation       : {dev['scale']:.4f}")
    print("=" * 60)


def run_pipeline(manifest_path, device="cuda:0", output_override=None) -> str:
    m = load_manifest(manifest_path)

    # (1) degrade GS -> world point cloud
    gs = read_gaussian_ply(m.gs_ply)
    ex_xyz, ex_rgb = degrade_gaussians_to_points(gs, m.opacity_threshold)
    print(f"[novel_pos] degraded GS: {len(ex_xyz)} points kept "
          f"(opacity>={m.opacity_threshold}) from {len(gs['xyz'])} gaussians")

    # (2) model
    model = load_mast3r(device=device)
    model.eval()

    # (3) operating pose = render pose (used directly in v1)
    T_render = sim3_from_pose_spec(m.novel_pose, device=device)

    # (4) frames both at the render pose (needed for the experiment's identity baseline)
    render_rgb = load_rgb01(m.render_image)
    comp_rgb = load_rgb01(m.completion_image)
    render_frame = build_frame(0, render_rgb, T_render, device=device)
    comp_frame = build_frame(1, comp_rgb, T_render, device=device)

    # (5) accuracy experiment (report only)
    dev = run_pose_accuracy_experiment(model, render_frame, comp_frame, device)
    _print_pose_report(dev)

    # (6) point-cloud update -- fresh completion frame (the experiment mutated the other)
    comp_frame_u = build_frame(2, comp_rgb, T_render, device=device)
    out_xyz, out_rgb, stats = update_point_cloud(
        model, comp_frame_u, T_render, ex_xyz, ex_rgb,
        conf_percentile=m.update_conf_percentile, device=device,
    )
    print(f"[novel_pos] update: +{stats['n_new']} new pts "
          f"(scale={stats['scale']:.4f}), total={stats['n_total']}")

    # (7) output merged point cloud
    out_path = output_override or m.output_ply
    write_ply(out_path, out_xyz, out_rgb)
    print(f"[novel_pos] wrote {out_path} ({stats['n_total']} points)")
    return str(out_path)
