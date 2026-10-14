"""Single-shot orchestration for the novel-pos pipeline (ADR-001 修订2).

Flow: load manifest -> degrade GS -> load model -> build render/completion frames (identity
pose; C makes the external pose geometrically irrelevant) -> mono-infer render -> gather the
render world pointmap onto the MASt3R grid -> fit similarity C (reflection allowed) ->
validation gate (report-only) -> apply C to the completion -> append -> write merged PLY.

WARNING: imports the CUDA backend (via the submodules below). GPU-only.
"""
import numpy as np
import lietorch

from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, resize_img

from mast3r_slam.novel_pos import align, render_io
from mast3r_slam.novel_pos.manifest import load_manifest
from mast3r_slam.novel_pos.gs_io import (
    read_gaussian_ply,
    degrade_gaussians_to_points,
    write_ply,
)
from mast3r_slam.novel_pos.pose_utils import sim3_from_pose_spec
from mast3r_slam.novel_pos.frames import load_rgb01, build_frame
from mast3r_slam.novel_pos.pose_experiment import run_validation_gate
from mast3r_slam.novel_pos.pc_update import update_point_cloud

MIN_FIT_PIXELS = 300  # floor on covered render pixels used to fit C


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

    # (3) frames at IDENTITY -- geometry comes from C, not the external pose
    T_build = lietorch.Sim3.Identity(1, device=device)
    render_rgb = load_rgb01(m.render_image)
    comp_rgb = load_rgb01(m.completion_image)
    render_frame = build_frame(0, render_rgb, T_build, device=device)
    comp_frame = build_frame(1, comp_rgb, T_build, device=device)
    if m.novel_pose is not None:
        _ = sim3_from_pose_spec(m.novel_pose, device=device)  # parsed for sanity only
        print(f"[novel_pos] (sanity) manifest render pose t={m.novel_pose.translation} "
              f"q={m.novel_pose.quaternion_xyzw}")

    # (4) resize transform for the gather (pure PIL/numpy; mast3r_utils already imported)
    _, transform = resize_img(render_rgb, 512, return_transformation=True)
    true_shape = tuple(int(x) for x in render_frame.img_true_shape.view(-1).tolist())  # (H, W)

    # (5) render mono inference ONCE (reused by the gate) -> X_r_cam
    Xr, Cr = mast3r_inference_mono(model, render_frame)
    render_frame.update_pointmap(Xr, Cr)
    X_r_cam = Xr.detach().cpu().numpy()                      # (HW, 3)

    # (6) gather the render world pointmap + coverage onto the MASt3R grid
    pm0 = render_io.load_pointmap(m.render_pointmap)         # (H0, W0, 3) GS world
    cov0 = render_io.load_coverage(m.render_coverage)        # (H0, W0) bool
    X_r_GS = render_io.gather_to_grid(pm0, true_shape, transform)                       # (HW, 3)
    cov = render_io.gather_to_grid(cov0[..., None].astype(np.float32), true_shape, transform)
    cov = cov.reshape(-1) > 0.5                              # (HW,)

    # (7) fit C (similarity WITH reflection) on covered, finite pixels
    mask = cov & np.isfinite(X_r_GS).all(1) & np.isfinite(X_r_cam).all(1)
    n_fit = int(mask.sum())
    if n_fit < MIN_FIT_PIXELS:
        raise RuntimeError(
            f"fit C: too few covered/finite pixels ({n_fit} < {MIN_FIT_PIXELS}); "
            f"check render_coverage and render_pointmap."
        )
    s, R, t, info = align.umeyama_with_reflection(X_r_cam[mask], X_r_GS[mask])
    refl = " (REFLECTION)" if info["det_R"] < 0 else ""
    print(f"[novel_pos] fit C: N={n_fit} s={s:.4f} det(R)={info['det_R']:+.3f} "
          f"rms={info['rms']:.4g}{refl}"
          + ("  [WARN degenerate scale]" if info["degenerate"] else ""))

    # (8) validation gate (report-only; render already inferred)
    gate = run_validation_gate(model, render_frame, comp_frame, device, m.gate)
    _print_gate_report(gate)
    if strict_gate and not gate.get("within_tolerance", False):
        raise RuntimeError(
            f"validation gate exceeded tolerance (strict mode): {gate}"
        )

    # (9) apply C to the completion and append
    out_xyz, out_rgb, stats = update_point_cloud(
        model, comp_frame, s, R, t, ex_xyz, ex_rgb,
        conf_percentile=m.update_conf_percentile, device=device,
    )
    print(f"[novel_pos] update: +{stats['n_new']} new pts "
          f"(s={stats['scale']:.4f}, det(R)={stats['det_R']:+.3f}), total={stats['n_total']}")

    # (10) output merged point cloud
    out_path = output_override or m.output_ply
    write_ply(out_path, out_xyz, out_rgb)
    print(f"[novel_pos] wrote {out_path} ({stats['n_total']} points)")
    return str(out_path)
