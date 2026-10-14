"""Backend-free smoke test for the novel-pos pure modules (ADR-001 修订3).

Runs WITHOUT a GPU, the MASt3R model, or the compiled CUDA backend. Needs only numpy
(+ plyfile for the gs_io round-trip; + torch & lietorch for the pose section, which is
skipped if unavailable).

    python -m mast3r_slam.novel_pos.smoke_test
"""
import pathlib
import tempfile

import numpy as np

from mast3r_slam.novel_pos import gs_io, align, gs_project


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
    assert len(pts) == 3, f"expected 3 kept points, got {len(pts)}"
    assert cols.dtype == np.uint8 and cols.shape == (3, 3), cols.shape
    assert int(cols[0, 0]) == 128, f"f_dc=0 -> 0.5 -> round(127.5)=128, got {cols[0, 0]}"

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "out.ply"
        gs_io.write_ply(p, pts, cols)
        from plyfile import PlyData
        assert len(PlyData.read(str(p))["vertex"].data) == 3
    print("OK: gs degrade + ply roundtrip")


# -------------------------------------------------------------------------- umeyama

def _random_rotation(rng):
    Q, _ = np.linalg.qr(rng.randn(3, 3))
    if np.linalg.det(Q) < 0:        # force a proper rotation (det +1)
        Q[:, 0] = -Q[:, 0]
    return Q


def test_umeyama_rotation_scale_translation():
    rng = np.random.RandomState(1)
    src = rng.randn(200, 3)
    s0, R0, t0 = 2.3, _random_rotation(rng), rng.randn(3)
    tgt = s0 * (src @ R0.T) + t0
    s, R, t, info = align.umeyama_with_reflection(src, tgt)
    assert abs(s - s0) < 1e-6 and np.allclose(R, R0, atol=1e-6), (s, info)
    assert np.allclose(t, t0, atol=1e-5) and info["det_R"] > 0 and info["rms"] < 1e-6, info
    print("OK: umeyama rotation+scale+translation")


def test_umeyama_known_reflection():
    rng = np.random.RandomState(2)
    src = rng.randn(200, 3)
    R0 = _random_rotation(rng).copy()
    R0[:, 2] = -R0[:, 2]            # make det(R0) = -1 (a reflection)
    s0, t0 = 1.7, rng.randn(3)
    tgt = s0 * (src @ R0.T) + t0
    s, R, t, info = align.umeyama_with_reflection(src, tgt)
    assert np.allclose(R, R0, atol=1e-6), R - R0          # SO(3)-forced solver would FAIL
    assert info["det_R"] < 0 and abs(s - s0) < 1e-6 and info["rms"] < 1e-6, info
    print("OK: umeyama recovers a known reflection")


def test_umeyama_guards():
    for bad, why in [
        ((np.zeros((2, 3)), np.zeros((2, 3))), "N<3"),
        ((np.ones((5, 3)), np.random.RandomState(3).randn(5, 3)), "zero source variance"),
    ]:
        try:
            align.umeyama_with_reflection(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {why}")
    print("OK: umeyama guards (N<3, degenerate source)")


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
    R = np.eye(3)
    t = np.zeros(3)
    xyz = np.array([
        [0.0, 0.0, 1.0],    # -> (50,50), Z=1 (nearest)
        [0.0, 0.0, 2.0],    # -> (50,50), Z=2 (farther, must lose to the above)
        [0.1, 0.0, 1.0],    # -> (60,50)
        [0.0, 0.0, -1.0],   # behind camera -> dropped
    ])
    X, cov = gs_project.project_gs_to_grid(xyz, R, t, K, (H, W))
    p_center = 50 * W + 50
    p_right = 50 * W + 60
    assert cov[p_center] and cov[p_right]
    assert np.allclose(X[p_center], [0.0, 0.0, 1.0]), X[p_center]   # nearest wins (z-buffer)
    assert np.allclose(X[p_right], [0.1, 0.0, 1.0]), X[p_right]
    assert cov.sum() == 2 and np.isnan(X[0]).all()                 # behind-camera dropped
    print("OK: project_gs_to_grid (pinhole + z-buffer + front culling)")


# -------------------------------------------------------------------------- pose_utils

def test_pose_utils():
    try:
        import torch  # noqa: F401
        import lietorch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIP pose_utils (torch/lietorch unavailable: {e})")
        return
    from mast3r_slam.novel_pos.pose_utils import decompose_relative_sim3
    rep = decompose_relative_sim3(lietorch.Sim3.Identity(1, device="cpu"))
    assert abs(rep["rotation_deg"]) < 1e-4 and abs(rep["scale"] - 1.0) < 1e-6, rep
    assert rep["translation_norm"] < 1e-6, rep
    print("OK: pose_utils identity")


if __name__ == "__main__":
    test_gs_degrade_and_ply_roundtrip()
    test_umeyama_rotation_scale_translation()
    test_umeyama_known_reflection()
    test_umeyama_guards()
    test_quat_wxyz_to_matrix()
    test_project_gs_to_grid()
    test_pose_utils()
    print("smoke test done")
