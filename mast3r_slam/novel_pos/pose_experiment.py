"""Validation gate (ADR-001 修订2): how well does the AI completion image's content
still agree with the render image's geometry.

Report-only: it NEVER changes the fitted transform C or the written points — the pipeline
may abort on it only under ``--strict-gate``. Reuses FrameTracker's no-calib relative-Sim3
solve, mirroring the no-calib branch of tracker.track (tracker.py:28-127).

WARNING: imports the CUDA backend (via tracker/mast3r_utils). GPU-only.
"""
import math

import torch

from mast3r_slam.config import config
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.mast3r_utils import mast3r_inference_mono, mast3r_match_asymmetric

from mast3r_slam.novel_pos.frames import KeyframeStub
from mast3r_slam.novel_pos.pose_utils import decompose_relative_sim3


def run_validation_gate(model, render_frame, completion_frame, device, thresholds) -> dict:
    """Match completion<->render and solve their relative Sim3 (ideal == identity).

    Assumes the pipeline has already mono-inferred the render frame (feat + X_canon set);
    only the completion<->render asymmetric match runs here. Both frames must share the
    same build pose so the solved relative Sim3 is the pure content deviation.

    Args:
        thresholds: GateThresholds with max_rotation_deg / max_translation_norm / max_log_scale.
    Returns {ok, match_frac, [rotation_deg, translation_norm, scale, within_tolerance] | reason}.
    """
    cfg = config["tracking"]

    # render should already be inferred by the pipeline; infer defensively if not.
    if render_frame.feat is None or render_frame.X_canon is None:
        Xr, Cr = mast3r_inference_mono(model, render_frame)
        render_frame.update_pointmap(Xr, Cr)

    idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
        model, completion_frame, render_frame, idx_i2j_init=None
    )
    idx_f2k = idx_f2k[0]
    valid_match_k = valid_match_k[0]
    Qk = torch.sqrt(Qff[idx_f2k] * Qkf)
    completion_frame.update_pointmap(Xff, Cff)

    tracker = FrameTracker(model, KeyframeStub(render_frame), device)
    img_size = completion_frame.img.shape[-2:]
    Xf, Xk, T_WCf, T_WCk, Cf, Ck, _, _ = tracker.get_points_poses(
        completion_frame, render_frame, idx_f2k, img_size, use_calib=False, K=None
    )

    valid_opt = (
        valid_match_k
        & (Cf > cfg["C_conf"])
        & (Ck > cfg["C_conf"])
        & (Qk > cfg["Q_conf"])
    )
    match_frac = float(valid_opt.sum() / valid_opt.numel())
    if match_frac < cfg["min_match_frac"]:
        return {"ok": False, "reason": "insufficient_matches", "match_frac": match_frac,
                "within_tolerance": False}

    try:
        _, T_CkCf = tracker.opt_pose_ray_dist_sim3(Xf, Xk, T_WCf, T_WCk, Qk, valid_opt)
    except Exception as e:  # noqa: BLE001 (mirror tracker.track's broad guard)
        return {"ok": False, "reason": f"solve_failed: {e}", "match_frac": match_frac,
                "within_tolerance": False}

    report = decompose_relative_sim3(T_CkCf)
    within = (
        report["rotation_deg"] <= thresholds.max_rotation_deg
        and report["translation_norm"] <= thresholds.max_translation_norm
        and abs(math.log(max(report["scale"], 1e-9))) <= thresholds.max_log_scale
    )
    report.update({"ok": True, "match_frac": match_frac, "within_tolerance": bool(within)})
    return report
