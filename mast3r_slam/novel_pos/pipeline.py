"""Single-shot orchestration for the novel-pos pipeline (ADR-001 修订3).

Flow: load manifest -> degrade GS -> load model -> build render/completion frames (identity
pose; MASt3R is pose-independent) -> mono-infer render -> PROJECT the degraded GS into the
render view (pose+K, z-buffer) to get the GS-world pointmap X_r_GS + coverage -> fit
similarity C (reflection allowed) X_r_GS ~= C.X_r_cam -> validation gate (report-only) ->
apply C to the completion -> append -> write merged PLY.

WARNING: imports the CUDA backend (via the submodules below). GPU-only.
"""
import numpy as np
import lietorch

from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, resize_img

from mast3r_slam.novel_pos import align, gs_project
from mast3r_slam.novel_pos.manifest import load_manifest
from mast3r_slam.novel_pos.gs_io import (
    read_gaussian_ply,
    degrade_gaussians_to_points,
    write_ply,
)
from mast3r_slam.novel_pos.frames import load_rgb01, build_frame
from mast3r_slam.novel_pos.pose_experiment import run_validation_gate
from mast3r_slam.novel_pos.pc_update import update_point_cloud

MIN_FIT_PIXELS = 300  # floor on covered (projected) pixels used to fit C


def _print_gate_report(dev: dict) -> None:
    print("=" * 60)
    print("Validation Gate (report only -- render<->completion content deviation)")
    if not dev.get("ok"):
        print(f"  FAILED: {dev.get('reason')}  (match_frac={dev.get('match_frac')})")
    else:
        flag = "OK" if dev["within_tolerance"] else "OUT OF TOLERANCE"
        print(f"  [{flag}] matches used     : {dev['match_frac'] * 100:.1f}% of pixels")
        print(f"  rotation deviation     : {dev['rotation_deg']:.3f} deg")
        print(f"  translation deviation  : {dev['translation_norm']:.4f}")
        print(f"  scale deviation        : {dev['scale']:.4f}")
    print("=" * 60)


def run_pipeline(manifest_path, device="cuda:0", output_override=None, strict_gate=False) -> str:
    m = load_manifest(manifest_path)

    # (1) degrade GS -> world point cloud
    gs = read_gaussian_ply(m.gs_ply)
    ex_xyz, ex_rgb = degrade_gaussians_to_points(gs, m.opacity_threshold)
    print(f"[novel_pos] degraded GS: {len(ex_xyz)} points kept "
          f"(opacity>={m.opacity_threshold}) from {len(gs['xyz'])} gaussians")

    # (2) model
    model = load_mast3r(device=device)
    model.eval()

    # (3) frames at IDENTITY -- MASt3R pointmaps are in the camera frame regardless of pose
    T_build = lietorch.Sim3.Identity(1, device=device)
    render_rgb = load_rgb01(m.render_image)
    comp_rgb = load_rgb01(m.completion_image)
    render_frame = build_frame(0, render_rgb, T_build, device=device)
    comp_frame = build_frame(1, comp_rgb, T_build, device=device)

    # (4) resize transform + MASt3R grid (pure PIL/numpy; mast3r_utils already imported)
    _, transform = resize_img(render_rgb, 512, return_transformation=True)
    true_shape = tuple(int(x) for x in render_frame.img_true_shape.view(-1).tolist())  # (H, W)
    rh, rw = render_rgb.shape[:2]
    if (m.intrinsics.width, m.intrinsics.height) != (rw, rh):
        print(f"[novel_pos] WARNING: intrinsics ({m.intrinsics.width}x{m.intrinsics.height}) "
              f"!= render_image ({rw}x{rh}); K_frame scale assumes render_image resolution.")

    # (5) render mono inference ONCE (reused by the gate) -> X_r_cam
    Xr, Cr = mast3r_inference_mono(model, render_frame)
    render_frame.update_pointmap(Xr, Cr)
    X_r_cam = Xr.detach().cpu().numpy()                      # (HW, 3)

    # (6) project the degraded GS into the render view -> X_r_GS + coverage (GS world)
    R_c2w = gs_project.quat_wxyz_to_matrix(m.novel_pose.quaternion_wxyz)
    t = np.asarray(m.novel_pose.translation, dtype=np.float64)
    K_frame = gs_project.intrinsics_to_K_frame(
        m.intrinsics.fx, m.intrinsics.fy, m.intrinsics.cx, m.intrinsics.cy, transform)
    X_r_GS, cov = gs_project.project_gs_to_grid(ex_xyz, R_c2w, t, K_frame, true_shape)
    print(f"[novel_pos] projected GS: coverage {int(cov.sum())}/{cov.size} pixels")

    # (7) fit C (similarity WITH reflection) on covered, finite pixels
    mask = cov & np.isfinite(X_r_GS).all(1) & np.isfinite(X_r_cam).all(1)
    n_fit = int(mask.sum())
    if n_fit < MIN_FIT_PIXELS:
        raise RuntimeError(
            f"fit C: too few covered pixels ({n_fit} < {MIN_FIT_PIXELS}); "
            f"check pose/intrinsics convention (wxyz/c2w/OpenCV) and GS density."
        )
    s, R, t_c, info = align.umeyama_with_reflection(X_r_cam[mask], X_r_GS[mask])
    refl = " (REFLECTION)" if info["det_R"] < 0 else ""
    print(f"[novel_pos] fit C: N={n_fit} s={s:.4f} det(R)={info['det_R']:+.3f} "
          f"rms={info['rms']:.4g}{refl}"
          + ("  [WARN degenerate scale]" if info["degenerate"] else ""))

    # (8) validation gate (report-only; render already inferred)
    gate = run_validation_gate(model, render_frame, comp_frame, device, m.gate)
    _print_gate_report(gate)
    if strict_gate and not gate.get("within_tolerance", False):
        raise RuntimeError(f"validation gate exceeded tolerance (strict mode): {gate}")

    # (9) apply C to the completion and append
    out_xyz, out_rgb, stats = update_point_cloud(
        model, comp_frame, s, R, t_c, ex_xyz, ex_rgb,
        conf_percentile=m.update_conf_percentile, device=device,
    )
    print(f"[novel_pos] update: +{stats['n_new']} new pts "
          f"(s={stats['scale']:.4f}, det(R)={stats['det_R']:+.3f}), total={stats['n_total']}")

    # (10) output merged point cloud
    out_path = output_override or m.output_ply
    write_ply(out_path, out_xyz, out_rgb)
    print(f"[novel_pos] wrote {out_path} ({stats['n_total']} points)")
    return str(out_path)
