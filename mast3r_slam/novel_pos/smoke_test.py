"""Backend-free smoke test for the novel-pos pure modules (ADR-001, asymmetric-only).

Runs WITHOUT a GPU, the MASt3R model, or the compiled CUDA backend. Needs only numpy
(+ plyfile for the gs_io round-trip).

    python -m mast3r_slam.novel_pos.smoke_test
"""
import pathlib
import tempfile

import numpy as np

from mast3r_slam.novel_pos import gs_io, align, gs_project
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
    print("smoke test done")
