"""Pose-accuracy EXPERIMENT (ADR-001).

Measures how well the AI completion image's pose agrees with the known render pose.
Report-only: it NEVER gates the point-cloud update. Reuses FrameTracker's no-calib
relative-Sim3 Gauss-Newton solve, mirroring the no-calib branch of tracker.track
(tracker.py:28-127) and get_points_poses (tracker.py:129-154).

WARNING: imports the CUDA backend (via tracker/mast3r_utils). GPU-only.
"""
import torch

from mast3r_slam.config import config
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.mast3r_utils import mast3r_inference_mono, mast3r_match_asymmetric

from mast3r_slam.novel_pos.frames import KeyframeStub
from mast3r_slam.novel_pos.pose_utils import decompose_relative_sim3


def run_pose_accuracy_experiment(model, render_frame, completion_frame, device) -> dict:
    """Both frames must be built at the SAME (render) pose, so the solved relative Sim3
    `T_CkCf` equals the completion-vs-render content deviation (ideal == identity).

    Returns a dict: {ok, match_frac, [rotation_deg, translation_norm, scale] | reason}.
    """
    cfg = config["tracking"]

    # 1) render pointmap (keyframe side); also populates render_frame.feat
    Xr, Cr = mast3r_inference_mono(model, render_frame)
    render_frame.update_pointmap(Xr, Cr)

    # 2) asymmetric match: frame_i = completion, frame_j = render (mirrors tracker.track)
    idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
        model, completion_frame, render_frame, idx_i2j_init=None
    )
    idx_f2k = idx_f2k[0]
    valid_match_k = valid_match_k[0]
    Qk = torch.sqrt(Qff[idx_f2k] * Qkf)
    completion_frame.update_pointmap(Xff, Cff)

    # 3) replicate the no-calib get_points_poses + valid mask (tracker.py:54-66, 129-154)
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
        return {"ok": False, "reason": "insufficient_matches", "match_frac": match_frac}

    # 4) solve the relative Sim3 (no-calib). Both frames share the render pose, so the
    #    returned T_CkCf is the measured deviation. Cholesky can fail -> report, don't crash.
    try:
        _, T_CkCf = tracker.opt_pose_ray_dist_sim3(Xf, Xk, T_WCf, T_WCk, Qk, valid_opt)
    except Exception as e:  # noqa: BLE001 (mirror tracker.track's broad guard)
        return {"ok": False, "reason": f"solve_failed: {e}", "match_frac": match_frac}

    report = decompose_relative_sim3(T_CkCf)
    report.update({"ok": True, "match_frac": match_frac})
    return report
