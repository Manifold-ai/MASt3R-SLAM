"""Backend-free smoke test for the novel-pos pure modules (ADR-001, asymmetric-only).

Runs WITHOUT a GPU, the MASt3R model, or the compiled CUDA backend. Needs only numpy
(+ plyfile for the gs_io round-trip).

    python -m mast3r_slam.novel_pos.smoke_test
"""
import pathlib
import tempfile

import numpy as np

from mast3r_slam.novel_pos import gs_io, align, gs_project, visibility
from mast3r_slam.novel_pos.pose_experiment import residual_gate
from mast3r_slam.novel_pos.pc_update import update_point_cloud
from mast3r_slam.novel_pos.manifest import GateThresholds


# ----------------------------------------------------------------------------- gs_io

def test_gs_degrade_and_ply_roundtrip():
    n = 5
    f_dc = np.zeros((n, 3), np.float32)                              # f_dc=0 -> color 0.5
    opacity = np.array([10.0, 10.0, -10.0, 10.0, -10.0], np.float32)  # keep 3 (sigmoid>0.5)
    gs = {
        "xyz": np.random.RandomState(0).randn(n, 3).astype(np.float32),
        "f_dc": f_dc,
        "opacity_logit": opacity,
    }
    pts, cols = gs_io.degrade_gaussians_to_points(gs, opacity_threshold=0.5)
    assert len(pts) == 3 and int(cols[0, 0]) == 128, (len(pts), cols[0, 0])
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "out.ply"
        gs_io.write_ply(p, pts, cols)
        from plyfile import PlyData
        assert len(PlyData.read(str(p))["vertex"].data) == 3
    print("OK: gs degrade + ply roundtrip")


# -------------------------------------------------------------------------- umeyama

def _random_rotation(rng):
    Q, _ = np.linalg.qr(rng.randn(3, 3))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_umeyama_rotation_scale_translation():
    rng = np.random.RandomState(1)
    src = rng.randn(200, 3)
    s0, R0, t0 = 2.3, _random_rotation(rng), rng.randn(3)
    s, R, t, info = align.umeyama_with_reflection(src, s0 * (src @ R0.T) + t0)
    assert abs(s - s0) < 1e-6 and np.allclose(R, R0, atol=1e-6), (s, info)
    assert np.allclose(t, t0, atol=1e-5) and info["det_R"] > 0 and info["rms"] < 1e-6, info
    print("OK: umeyama rotation+scale+translation")


def test_umeyama_known_reflection():
    rng = np.random.RandomState(2)
    src = rng.randn(200, 3)
    R0 = _random_rotation(rng).copy()
    R0[:, 2] = -R0[:, 2]            # det(R0) = -1
    s0, t0 = 1.7, rng.randn(3)
    s, R, t, info = align.umeyama_with_reflection(src, s0 * (src @ R0.T) + t0)
    assert np.allclose(R, R0, atol=1e-6) and info["det_R"] < 0, (R - R0, info["det_R"])
    assert abs(s - s0) < 1e-6 and info["rms"] < 1e-6, info
    print("OK: umeyama recovers a known reflection")


def test_umeyama_guards():
    for bad in [(np.zeros((2, 3)), np.zeros((2, 3))),
                (np.ones((5, 3)), np.random.RandomState(3).randn(5, 3))]:
        try:
            align.umeyama_with_reflection(*bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    print("OK: umeyama guards (N<3, degenerate source)")


def test_umeyama_weighted_downweights_outliers():
    # 200 inliers follow y=s0·x+t0; 60 outliers map to garbage. Weighting the outliers near
    # zero must recover the inlier transform, where the unweighted fit is pulled off.
    rng = np.random.RandomState(7)
    src_in = rng.randn(200, 3)
    s0, R0, t0 = 2.0, np.eye(3), np.array([1.0, 0, 0])
    tgt_in = s0 * (src_in @ R0.T) + t0
    src_out, tgt_out = rng.randn(60, 3), rng.randn(60, 3) * 20
    src = np.vstack([src_in, src_out])
    tgt = np.vstack([tgt_in, tgt_out])
    w = np.concatenate([np.ones(200), np.full(60, 1e-3)])
    s, _, _, _ = align.umeyama_with_reflection(src, tgt, weights=w)
    su, _, _, _ = align.umeyama_with_reflection(src, tgt)          # unweighted -> biased
    assert abs(s - s0) < 0.05 and abs(s - s0) < abs(su - s0), (s, su)
    print("OK: umeyama weighted down-weights outliers")


# ------------------------------------------------------------------------- gs_project

def test_quat_wxyz_to_matrix():
    assert np.allclose(gs_project.quat_wxyz_to_matrix([1, 0, 0, 0]), np.eye(3))
    c = np.cos(np.pi / 4)
    R = gs_project.quat_wxyz_to_matrix([c, 0, 0, c])             # 90 deg about +z
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9), R
    print("OK: quat_wxyz_to_matrix (identity + 90deg z)")


def test_project_gs_to_grid():
    H, W = 100, 100
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    xyz = np.array([
        [0.0, 0.0, 1.0],    # -> (50,50), Z=1 (nearest)
        [0.0, 0.0, 2.0],    # -> (50,50), Z=2 (must lose to the above)
        [0.1, 0.0, 1.0],    # -> (60,50)
        [0.0, 0.0, -1.0],   # behind camera -> dropped
    ])
    X, cov = gs_project.project_gs_to_grid(xyz, np.eye(3), np.zeros(3), K, (H, W))
    assert cov[50 * W + 50] and cov[50 * W + 60] and cov.sum() == 2, cov.sum()
    assert np.allclose(X[50 * W + 50], [0.0, 0.0, 1.0]), X[50 * W + 50]   # z-buffer: nearest wins
    assert np.allclose(X[50 * W + 60], [0.1, 0.0, 1.0]), X[50 * W + 60]
    print("OK: project_gs_to_grid (pinhole + z-buffer + front culling)")


# ------------------------------------------------------------------ residual gate

def test_residual_gate():
    rng = np.random.RandomState(4)
    X = rng.randn(200, 3) * 2 + np.array([0.0, 0.0, 5.0])   # depth ~5
    v = np.ones(200, bool)
    th = GateThresholds(max_rel_residual=0.05)
    g = residual_gate(X, X.copy(), v, th)
    assert g["ok"] and g["within_tolerance"] and g["rel_residual"] < 1e-9, g
    g2 = residual_gate(X, X + np.array([1.0, 0, 0]), v, th)   # shift 1, depth ~5 -> rel ~0.2
    assert g2["ok"] and not g2["within_tolerance"] and g2["rel_residual"] > 0.1, g2
    assert not residual_gate(X, X, np.zeros(200, bool), th)["ok"]   # no anchor
    print("OK: residual_gate")


def test_residual_gate_anchor_mask():
    rng = np.random.RandomState(5)
    X = rng.randn(400, 3) * 2 + np.array([0.0, 0.0, 5.0])
    Xc = X.copy()
    Xc[:200] += np.array([1.0, 0, 0])                         # first half moved by the completion
    v = np.ones(400, bool)
    v[200:250] = False                                        # unmatched inside V
    Xc[260] = np.nan                                          # non-finite inside V
    th = GateThresholds(max_rel_residual=0.05)
    keep = np.arange(400) >= 200                              # V drops the moved half
    g_all = residual_gate(X, Xc, v, th)
    g_vis = residual_gate(X, Xc, v, th, anchor_mask=keep)
    assert g_vis["ok"] and g_vis["median_residual"] < 1e-9 < g_all["median_residual"], (g_vis, g_all)
    assert abs(g_vis["frac_anchor"] - 149 / 400) < 1e-12, g_vis   # V intersects, not replaces
    assert residual_gate(X, Xc, v, th, anchor_mask=None) == g_all          # None == no mask
    few = np.zeros(400, bool)
    few[:50] = True
    assert residual_gate(X, Xc, v, th, anchor_mask=few)["reason"] == "insufficient_anchor"
    print("OK: residual_gate anchor_mask")


# ----------------------------------------------------------------- visibility (C-002)

def test_vis_mode_and_threshold():
    assert visibility.resolve_vis_mode("auto", True) == "mask"
    assert visibility.resolve_vis_mode("auto", False) == "off"
    assert visibility.resolve_vis_mode("off", True) == "off"
    assert visibility.resolve_vis_mode("weight", True) == "weight"
    for mode, has_map in [("mask", False), ("weight", False), ("bogus", True)]:
        try:
            visibility.resolve_vis_mode(mode, has_map)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {mode}, has_map={has_map}")
    assert visibility.check_threshold(1.0) == 1.0 and visibility.check_threshold(0.5) == 0.5
    for tau in [0.0, 1.5, -0.1, float("nan")]:
        try:
            visibility.check_threshold(tau)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for tau={tau}")
    print("OK: visibility mode + threshold")


def test_vis_as_gray_u8():
    g = (np.arange(12, dtype=np.uint8) * 20).reshape(3, 4)
    assert np.array_equal(visibility.as_gray_u8(g, (3, 4)), g)
    assert np.array_equal(visibility.as_gray_u8(np.dstack([g, g, g]), (3, 4)), g)
    g2 = g.copy()
    g2[0, 0] += 1
    bad = [
        np.dstack([g, g, g2]),                     # 3 channels that differ
        np.dstack([g, g, g, g]),                   # 4 channels (RGBA)
        g.astype(np.uint16),                       # 16-bit
        g[:, :3],                                  # wrong size
    ]
    for img in bad:
        try:
            visibility.as_gray_u8(img, (3, 4))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for shape {img.shape} dtype {img.dtype}")
    print("OK: visibility as_gray_u8")


def test_vis_alpha_roundtrip_and_threshold():
    v = np.arange(256)
    a = v.astype(np.uint8).astype(np.float32) / 255.0      # what frames.load_visibility_alpha feeds
    assert np.array_equal(np.uint8(a * 255), v)              # resize_img entry: no truncation loss
    V = visibility.visible_set(a, 0.5)
    assert int(np.argmax(V)) == 128 and V[128:].all() and not V[:128].any(), np.argmax(V)
    V1 = visibility.visible_set(a, 1.0)
    assert V1.sum() == 1 and V1[255]                          # tau=1 -> only 255
    print("OK: visibility alpha round-trip + threshold")


def test_vis_fit_set_and_weights():
    mask = np.array([True, True, True, False, True])
    vis = np.array([True, False, True, True, False])
    m2, n_drop = visibility.restrict_fit_mask(mask, vis, "mask")
    assert np.array_equal(m2, [True, False, True, False, False]) and n_drop == 2, (m2, n_drop)
    for mode in ("off", "weight"):
        m3, n3 = visibility.restrict_fit_mask(mask, vis, mode)
        assert m3 is mask and n3 == 0
    alpha = np.array([1.0, 0.5, 0.0, 0.25])
    w, kept = visibility.alpha_weights(None, alpha)          # --fit-weight none -> w = alpha
    assert np.allclose(w, alpha) and abs(kept - 1.75 / 4) < 1e-12, (w, kept)
    fw = np.array([2.0, 2.0, 4.0, 0.0])                       # conf weights (last floored to 0)
    w, kept = visibility.alpha_weights(fw, alpha)
    assert np.allclose(w, [2.0, 1.0, 0.0, 0.0]) and abs(kept - 3.0 / 8.0) < 1e-12, (w, kept)
    print("OK: visibility fit set + alpha weights")


# -------------------------------------------------------------- pc_update (apply C)

def test_update_point_cloud():
    X = np.array([[1.0, 1, 1], [0, 0, 0], [2, 2, 2], [3, 3, 3]])
    C = np.array([0.1, 0.9, 0.5, 0.7])
    uimg = np.ones((4, 3)) * 0.5
    v = np.array([True, False, True, False])
    ex_xyz = np.zeros((2, 3), np.float32)
    ex_rgb = np.zeros((2, 3), np.uint8)
    s, R, t = 2.0, np.eye(3), np.array([1.0, 0, 0])   # [1,1,1] -> 2*[1,1,1]+[1,0,0]=[3,2,2]

    out_xyz, _, st = update_point_cloud(X, C, uimg, v, s, R, t, ex_xyz, ex_rgb, conf_percentile=0.0)
    assert st["n_existing"] == 2 and st["n_new"] == 4 and st["n_total"] == 6, st
    assert np.allclose(out_xyz[2], [3, 2, 2]), out_xyz[2]          # apply C correct
    _, _, st2 = update_point_cloud(X, C, uimg, v, s, R, t, ex_xyz, ex_rgb, conf_percentile=50.0)
    assert st2["n_new"] == 2, st2                                 # keep C >= median 0.6
    _, _, st3 = update_point_cloud(X, C, uimg, v, s, R, t, ex_xyz, ex_rgb, new_only=True)
    assert st3["n_new"] == 2 and st3["new_only"], st3             # keep ~vmatch
    print("OK: update_point_cloud (apply C + select)")


if __name__ == "__main__":
    test_gs_degrade_and_ply_roundtrip()
    test_umeyama_rotation_scale_translation()
    test_umeyama_known_reflection()
    test_umeyama_guards()
    test_umeyama_weighted_downweights_outliers()
    test_quat_wxyz_to_matrix()
    test_project_gs_to_grid()
    test_residual_gate()
    test_update_point_cloud()
    test_residual_gate_anchor_mask()
    test_vis_mode_and_threshold()
    test_vis_as_gray_u8()
    test_vis_alpha_roundtrip_and_threshold()
    test_vis_fit_set_and_weights()
    print("smoke test done")
