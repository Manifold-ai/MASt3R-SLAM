"""Single-shot orchestration for the novel-pos pipeline (ADR-001, asymmetric-only).

Flow: load manifest -> degrade GS -> load model -> build render/completion frames (identity
pose) -> ONE mast3r_match_asymmetric(render, completion) giving both pointmaps IN THE RENDER
FRAME + the match -> project the GS into the render view (pose+K, z-buffer) -> fit similarity
C (reflection allowed, render-confidence-weighted) X_r_GS ~= C.X_render -> residual gate ->
apply C to the completion pointmap -> select (conf percentile, or new_only via valid_match)
-> append -> write merged PLY. An optional render visibility map (C-002) masks or weights the
C fit and restricts the gate anchors to visible pixels.

WARNING: imports the CUDA backend (via the submodules below). GPU-only.
"""
import os

import numpy as np
import lietorch

from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_match_asymmetric, resize_img

from mast3r_slam.novel_pos import align, gs_project, visibility
from mast3r_slam.novel_pos.manifest import load_manifest
from mast3r_slam.novel_pos.gs_io import (
    read_gaussian_ply,
    degrade_gaussians_to_points,
    write_ply,
)
from mast3r_slam.novel_pos.frames import load_rgb01, build_frame, load_visibility_alpha
from mast3r_slam.novel_pos.pose_experiment import anchor_set, residual_gate
from mast3r_slam.novel_pos.pc_update import update_point_cloud

MIN_FIT_PIXELS = 300  # floor on covered (projected) pixels used to fit C


def _print_gate_report(gate: dict) -> None:
    print("=" * 60)
    print("Validation Gate (report only -- render<->completion geometric residual)")
    if not gate.get("ok"):
        print(f"  FAILED: {gate.get('reason')}  (anchor frac={gate.get('frac_anchor')})")
    else:
        flag = "OK" if gate["within_tolerance"] else "OUT OF TOLERANCE"
        print(f"  [{flag}] anchor pixels : {gate['frac_anchor'] * 100:.1f}%")
        print(f"  median residual      : {gate['median_residual']:.4g}")
        print(f"  relative residual    : {gate['rel_residual']:.4g}")
    print("=" * 60)


def _save_debug(debug_dir, render_frame, X_r_cam, X_r_GS, X_comp, render_conf, cov, eval_mask,
                fit_mask, vmatch, true_shape, s, R, t_c, info, gate, fit_weight="conf", fit_w=None,
                vis_mode="off", tau=0.5, vis_alpha=None, vis=None, n_anchor=0):
    """Dump C-fit inputs/result + the completion pointmap/match for debugging. Never raises.

    ``residual`` and the overlay PLYs use the fixed evaluation set ``eval_mask`` (coverage &
    finite), not the fit set, so runs with different visibility modes stay comparable (C-002).
    """
    try:
        os.makedirs(debug_dir, exist_ok=True)
        hw = X_r_cam.shape[0]
        fitted = (s * (X_r_cam @ np.asarray(R).T) + np.asarray(t_c).reshape(1, 3)).astype(np.float32)
        resid = np.full(hw, np.nan, dtype=np.float32)
        resid[eval_mask] = np.linalg.norm(fitted[eval_mask] - X_r_GS[eval_mask], axis=1)
        fit_w_full = np.zeros(hw, dtype=np.float32)   # per-pixel C-fit weight (0 off the fit set)
        fit_w_full[fit_mask] = np.asarray(fit_w, np.float32) if fit_w is not None else 1.0
        # map not loaded (off / no map): NaN alpha and an EMPTY V, never a silent "all visible"
        vis_loaded = vis is not None
        vis_alpha_full = (np.asarray(vis_alpha, np.float32) if vis_loaded
                          else np.full(hw, np.nan, dtype=np.float32))
        vis_mask = np.asarray(vis, bool) if vis_loaded else np.zeros(hw, dtype=bool)
        np.savez(
            os.path.join(debug_dir, "fit_debug.npz"),
            X_r_cam=X_r_cam.astype(np.float32), X_r_GS=X_r_GS.astype(np.float32),
            X_comp=np.asarray(X_comp, np.float32), vmatch=np.asarray(vmatch, bool),
            fitted=fitted, residual=resid, render_conf=render_conf,
            fit_weight=np.asarray(str(fit_weight)), fit_w=fit_w_full,
            coverage=cov, mask=eval_mask, fit_mask=fit_mask,
            true_shape=np.asarray(true_shape, np.int64),
            s=np.float64(s), R=np.asarray(R, np.float64), t=np.asarray(t_c, np.float64),
            det_R=np.float64(info["det_R"]), rms=np.float64(info["rms"]),
            n_fit=np.int64(int(fit_mask.sum())), degenerate=bool(info["degenerate"]),
            gate=np.asarray(str(gate)), n_anchor=np.int64(n_anchor),
            vis_mode=np.asarray(str(vis_mode)), vis_threshold=np.float64(tau),
            vis_loaded=np.bool_(vis_loaded), vis_alpha=vis_alpha_full, vis_mask=vis_mask,
        )
        col = (render_frame.uimg.reshape(-1, 3).cpu().numpy() * 255).round().astype(np.uint8)
        write_ply(os.path.join(debug_dir, "dbg_render_fitted.ply"), fitted[eval_mask], col[eval_mask])
        write_ply(os.path.join(debug_dir, "dbg_render_gs.ply"), X_r_GS[eval_mask], col[eval_mask])
        on_v = eval_mask & vis_mask
        on_v_note = f", on V={np.median(resid[on_v]):.4g}" if on_v.any() else ""
        print(f"[novel_pos] debug: wrote {debug_dir}/ "
              f"(fit_debug.npz + 2 overlay plys; residual median={np.nanmedian(resid):.4g}{on_v_note})")
    except Exception as e:  # noqa: BLE001 (debug must never break the run)
        print(f"[novel_pos] WARNING: debug save failed: {e}")


def run_pipeline(manifest_path, device="cuda:0", output_override=None, strict_gate=False,
                 debug_dir=None, new_only=False, fit_weight="conf", fit_conf_percentile=0.0,
                 vis_mode="auto", vis_threshold=0.5) -> str:
    m = load_manifest(manifest_path)

    # visibility map (C-002): validate the mode/threshold before any expensive step
    mode = visibility.resolve_vis_mode(vis_mode, m.visibility_map is not None)
    tau = visibility.check_threshold(vis_threshold)
    if m.visibility_map is not None and mode == "off":
        print("[novel_pos] visibility_map provided but --vis-mode off: ignored")
    elif mode != "off" and config["dataset"]["img_downsample"] != 1:
        raise RuntimeError("visibility map requires dataset.img_downsample == 1 (C-002 §三): "
                           "the map is aligned to the full-resolution MASt3R grid")

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

    # (4) resize transform + MASt3R grid
    _, transform = resize_img(render_rgb, 512, return_transformation=True)
    true_shape = tuple(int(x) for x in render_frame.img_true_shape.view(-1).tolist())  # (H, W)
    rh, rw = render_rgb.shape[:2]
    if (m.intrinsics.width, m.intrinsics.height) != (rw, rh):
        print(f"[novel_pos] WARNING: intrinsics ({m.intrinsics.width}x{m.intrinsics.height}) "
              f"!= render_image ({rw}x{rh}); K_frame scale assumes render_image resolution.")

    # (4b) optional visibility map -> alpha on the same grid (same resize/crop as the render)
    vis_alpha = vis = None
    if mode != "off":
        a_grid, a_shape = load_visibility_alpha(m.visibility_map, (rh, rw))
        if a_shape != true_shape:
            raise RuntimeError(f"visibility: aligned grid {a_shape} != render true_shape {true_shape}")
        vis_alpha = a_grid.reshape(-1)
        vis = visibility.visible_set(vis_alpha, tau)
        n_vis = int(vis.sum())
        if n_vis == 0:
            raise RuntimeError(f"visibility: V is empty (no pixel with alpha >= {tau}); "
                               f"check the map or lower --vis-threshold")
        note = ("  [all visible: mask == off]" if mode == "mask" and n_vis == vis.size else
                "  [alpha all 1: weight == off]" if mode == "weight" and (vis_alpha == 1).all()
                else "")
        print(f"[novel_pos] visibility: mode={mode} tau={tau:g} "
              f"visible={n_vis / vis.size:.1%} ({n_vis}/{vis.size}){note}")

    # (5) ONE asymmetric inference -> render + completion pointmaps, BOTH in the render frame
    (idx, valid_match, X_render_t, C_render_t, Q_render_t,
     X_comp_t, C_comp_t, Q_comp_t) = mast3r_match_asymmetric(model, render_frame, comp_frame)
    X_r_cam = X_render_t.detach().cpu().numpy()                 # (HW,3) render grid, render frame
    X_comp = X_comp_t.detach().cpu().numpy()                    # (HW,3) completion grid, render frame
    C_comp = C_comp_t.reshape(-1).detach().cpu().numpy()        # (HW,)
    vmatch = valid_match[0].reshape(-1).detach().cpu().numpy().astype(bool)  # (HW,) True=anchor
    render_conf = C_render_t.reshape(-1).detach().cpu().numpy()
    comp_uimg = comp_frame.uimg.detach().cpu().numpy()          # (H,W,3)
    if vis is not None and vis.size != X_r_cam.shape[0]:
        raise RuntimeError(f"visibility: {vis.size} pixels != pointmap {X_r_cam.shape[0]}; "
                           f"dataset.img_downsample must be 1 (C-002 §三)")

    # (6) project the degraded GS into the render view -> X_r_GS + coverage (GS world)
    R_c2w = gs_project.quat_wxyz_to_matrix(m.novel_pose.quaternion_wxyz)
    t = np.asarray(m.novel_pose.translation, dtype=np.float64)
    K_frame = gs_project.intrinsics_to_K_frame(
        m.intrinsics.fx, m.intrinsics.fy, m.intrinsics.cx, m.intrinsics.cy, transform)
    X_r_GS, cov = gs_project.project_gs_to_grid(ex_xyz, R_c2w, t, K_frame, true_shape)
    print(f"[novel_pos] projected GS: coverage {int(cov.sum())}/{cov.size} pixels")

    # (7) fit C (similarity WITH reflection) on covered, finite pixels, weighted by render conf.
    # eval_mask is the fixed evaluation set; the fit set additionally drops invisible pixels in
    # vis mask mode (C-002 §五).
    eval_mask = cov & np.isfinite(X_r_GS).all(1) & np.isfinite(X_r_cam).all(1)
    mask, n_vis_drop = visibility.restrict_fit_mask(eval_mask, vis, mode)
    n_fit = int(mask.sum())
    if n_fit < MIN_FIT_PIXELS:
        if n_fit + n_vis_drop >= MIN_FIT_PIXELS:           # the visibility mask caused it
            raise RuntimeError(
                f"fit C: too few covered pixels ({n_fit} < {MIN_FIT_PIXELS}) after visibility "
                f"mask (-{n_vis_drop}, tau={tau:g}); lower --vis-threshold, check the "
                f"visibility map, or use --vis-mode weight/off."
            )
        raise RuntimeError(
            f"fit C: too few covered pixels ({n_fit} < {MIN_FIT_PIXELS}); "
            f"check pose/intrinsics convention (wxyz/c2w/OpenCV) and GS density."
        )
    if fit_weight == "conf":
        # down-weight low-confidence correspondences: their MASt3R depth can disagree with the
        # GS and bias the whole-view least-squares C. conf is positive; clip defensively.
        fit_w = np.clip(render_conf[mask].astype(np.float64), 0.0, None)
        if fit_conf_percentile > 0:                        # optional hard floor: drop lowest P%
            thr = np.percentile(render_conf[mask], fit_conf_percentile)
            fit_w = np.where(render_conf[mask] >= thr, fit_w, 0.0)
        n_eff = int((fit_w > 0).sum())
        if n_eff < MIN_FIT_PIXELS:
            raise RuntimeError(
                f"fit C: too few weighted pixels ({n_eff} < {MIN_FIT_PIXELS}) after conf floor "
                f"{fit_conf_percentile:.0f}%; lower --fit-conf-percentile."
            )
    else:
        fit_w, n_eff = None, n_fit
    floor_note = (f", floor={fit_conf_percentile:.0f}%, n_eff={n_eff}"
                  if fit_weight == "conf" and fit_conf_percentile > 0 else "")
    vis_note = f", vis=mask -{n_vis_drop}px" if mode == "mask" else ""
    if mode == "weight":                                   # conf floor first, then multiply by alpha
        fit_w, kept = visibility.alpha_weights(fit_w, vis_alpha[mask])
        n_eff = int((fit_w > 0).sum())
        if n_eff < MIN_FIT_PIXELS:
            raise RuntimeError(
                f"fit C: too few weighted pixels ({n_eff} < {MIN_FIT_PIXELS}) after multiplying "
                f"by visibility alpha; check the visibility map."
            )
        vis_note = f", vis=weight w-kept={kept:.1%}, n_eff_alpha={n_eff}"
    s, R, t_c, info = align.umeyama_with_reflection(X_r_cam[mask], X_r_GS[mask], weights=fit_w)
    refl = " (REFLECTION)" if info["det_R"] < 0 else ""
    print(f"[novel_pos] fit C: N={n_fit} (weight={fit_weight}{floor_note}{vis_note}) "
          f"s={s:.4f} det(R)={info['det_R']:+.3f} rms={info['rms']:.4g}{refl}"
          + ("  [WARN degenerate scale]" if info["degenerate"] else ""))

    # (8) residual validation gate (report-only; both pointmaps already in render frame).
    # With a visibility map, anchors are restricted to the binary visible set V in both modes.
    gate = residual_gate(X_r_cam, X_comp, vmatch, m.gate, anchor_mask=vis)
    anchor = anchor_set(X_r_cam, X_comp, vmatch)
    n_anchor = int((anchor & vis).sum()) if vis is not None else int(anchor.sum())
    if vis is not None:
        print(f"[novel_pos] gate anchors: visibility dropped {int(anchor.sum()) - n_anchor} "
              f"of {int(anchor.sum())} (kept {n_anchor})")
    _print_gate_report(gate)
    if debug_dir:
        _save_debug(debug_dir, render_frame, X_r_cam, X_r_GS, X_comp, render_conf, cov, eval_mask,
                    mask, vmatch, true_shape, s, R, t_c, info, gate, fit_weight, fit_w,
                    mode, tau, vis_alpha, vis, n_anchor)
    if strict_gate and not gate.get("within_tolerance", False):
        raise RuntimeError(f"validation gate exceeded tolerance (strict mode): {gate}")

    # (9) apply C to the completion and append
    out_xyz, out_rgb, stats = update_point_cloud(
        X_comp, C_comp, comp_uimg, vmatch, s, R, t_c, ex_xyz, ex_rgb,
        conf_percentile=m.update_conf_percentile, new_only=new_only,
    )
    print(f"[novel_pos] update: +{stats['n_new']} new pts "
          f"(new_only={stats['new_only']}, s={stats['scale']:.4f}, det(R)={stats['det_R']:+.3f}), "
          f"total={stats['n_total']}")

    # (10) output merged point cloud
    out_path = output_override or m.output_ply
    write_ply(out_path, out_xyz, out_rgb)
    print(f"[novel_pos] wrote {out_path} ({stats['n_total']} points)")
    return str(out_path)
