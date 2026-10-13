"""Backend-free smoke test for the novel-pos pure modules (ADR-001).

Runs WITHOUT a GPU, the MASt3R model, or the compiled CUDA backend. Needs only
numpy + plyfile (+ pyyaml). The pose section additionally needs torch + lietorch and
is skipped if they are unavailable.

    python -m mast3r_slam.novel_pos.smoke_test
"""
import pathlib
import tempfile

import numpy as np

from mast3r_slam.novel_pos import gs_io


def test_gs_degrade_and_ply_roundtrip():
    n = 5
    f_dc = np.zeros((n, 3), np.float32)                         # f_dc=0 -> color 0.5
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

        back = PlyData.read(str(p))["vertex"].data
        assert len(back) == 3, len(back)
    print("OK: gs degrade + ply roundtrip")


def test_pose_utils():
    try:
        import torch  # noqa: F401
        import lietorch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"SKIP pose_utils (torch/lietorch unavailable: {e})")
        return
    from mast3r_slam.novel_pos.manifest import PoseSpec
    from mast3r_slam.novel_pos.pose_utils import (
        sim3_from_pose_spec,
        decompose_relative_sim3,
    )

    identity = PoseSpec(translation=[0.0, 0.0, 0.0], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0])
    T = sim3_from_pose_spec(identity, device="cpu")
    rep = decompose_relative_sim3(T)
    assert abs(rep["rotation_deg"]) < 1e-4, rep
    assert abs(rep["scale"] - 1.0) < 1e-6, rep
    assert rep["translation_norm"] < 1e-6, rep
    print("OK: pose_utils identity")


if __name__ == "__main__":
    test_gs_degrade_and_ply_roundtrip()
    test_pose_utils()
    print("smoke test done")
