"""Backend-free smoke test for the novel-pos pure modules (ADR-001 修订2).

Runs WITHOUT a GPU, the MASt3R model, or the compiled CUDA backend. Needs only numpy
(+ plyfile for the gs_io round-trip; + torch & lietorch for the pose section, which is
skipped if unavailable).

    python -m mast3r_slam.novel_pos.smoke_test
"""
import pathlib
import tempfile

import numpy as np

from mast3r_slam.novel_pos import gs_io, align, render_io


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
    assert abs(s - s0) < 1e-6, (s, s0)
    assert np.allclose(R, R0, atol=1e-6), R - R0
    assert np.allclose(t, t0, atol=1e-5), (t, t0)
    assert info["det_R"] > 0 and info["rms"] < 1e-6, info
    print("OK: umeyama rotation+scale+translation")


def test_umeyama_known_reflection():
    rng = np.random.RandomState(2)
    src = rng.randn(200, 3)
    R0 = _random_rotation(rng).copy()
    R0[:, 2] = -R0[:, 2]            # make det(R0) = -1 (a reflection)
    s0, t0 = 1.7, rng.randn(3)
    tgt = s0 * (src @ R0.T) + t0
    s, R, t, info = align.umeyama_with_reflection(src, tgt)
    assert np.allclose(R, R0, atol=1e-6), R - R0          # SO(3)-forced solver would FAIL here
    assert info["det_R"] < 0, info["det_R"]
    assert abs(s - s0) < 1e-6 and info["rms"] < 1e-6, info
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


# ---------------------------------------------------------------------- gather map

def test_gather_mapping():
    H0, W0 = 8, 12
    arr = (np.arange(H0)[:, None] * 100 + np.arange(W0)[None, :]).astype(np.float32)[..., None]
    transform = (2.0, 2.0, 1.0, 0.5)                      # (scale_w, scale_h, hcw, hch)
    true_shape = (3, 4)
    got = render_io.gather_to_grid(arr, true_shape, transform)
    assert got.shape == (12, 1), got.shape
    sw, sh, hcw, hch = transform
    exp = np.empty((3, 4, 1), np.float32)
    for v in range(3):
        for u in range(4):
            uo = min(int(round((u + hcw) * sw)), W0 - 1)
            vo = min(int(round((v + hch) * sh)), H0 - 1)
            exp[v, u, 0] = arr[vo, uo, 0]
    assert np.array_equal(got, exp.reshape(12, 1)), (got.ravel(), exp.ravel())

    # identity: scale 1, no crop, grid == original -> flatten
    ident = render_io.gather_to_grid(arr, (H0, W0), (1.0, 1.0, 0.0, 0.0))
    assert np.array_equal(ident, arr.reshape(H0 * W0, 1))
    print("OK: gather nearest mapping + identity")


# -------------------------------------------------------------------------- pose_utils

def test_pose_utils():
    try:
        import torch  # noqa: F401
        import lietorch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIP pose_utils (torch/lietorch unavailable: {e})")
        return
    from mast3r_slam.novel_pos.manifest import PoseSpec
    from mast3r_slam.novel_pos.pose_utils import sim3_from_pose_spec, decompose_relative_sim3

    identity = PoseSpec(translation=[0.0, 0.0, 0.0], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0])
    rep = decompose_relative_sim3(sim3_from_pose_spec(identity, device="cpu"))
    assert abs(rep["rotation_deg"]) < 1e-4 and abs(rep["scale"] - 1.0) < 1e-6, rep
    assert rep["translation_norm"] < 1e-6, rep
    print("OK: pose_utils identity")


if __name__ == "__main__":
    test_gs_degrade_and_ply_roundtrip()
    test_umeyama_rotation_scale_translation()
    test_umeyama_known_reflection()
    test_umeyama_guards()
    test_gather_mapping()
    test_pose_utils()
    print("smoke test done")
