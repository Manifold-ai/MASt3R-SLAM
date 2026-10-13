"""Pose helpers for novel-pos (ADR-001). Backend-free: torch + lietorch only.

lietorch.Sim3 raw data layout is [t(3), quat xyzw(4), scale(1)] (embedded_dim 8),
verified by repo usage (lietorch_utils.as_SE3 splits [3, 4, 1]; evaluate.py writes
x,y,z,qx,qy,qz,qw). The repo only ever touches `Sim3.data`, never `.scale()`/`.matrix()`,
so we do the same here.
"""
import math

import torch
import lietorch


def sim3_from_pose_spec(pose, device="cuda:0", dtype=torch.float32) -> "lietorch.Sim3":
    """Build a lietorch.Sim3 (world <- camera) from a manifest PoseSpec."""
    tx, ty, tz = pose.translation
    qx, qy, qz, qw = pose.quaternion_xyzw
    s = float(pose.scale)
    data = torch.tensor([tx, ty, tz, qx, qy, qz, qw, s], dtype=dtype, device=device)
    return lietorch.Sim3(data[None])  # batch dim -> (1, 8)


def decompose_relative_sim3(T_rel) -> dict:
    """Deviation-from-identity of a relative Sim3: rotation(deg) / translation / scale.

    Reads .data directly: [tx, ty, tz, qx, qy, qz, qw, s]. rotation angle = 2*acos(|qw|)
    (abs handles the quaternion double cover).
    """
    d = T_rel.data.detach().cpu().reshape(-1)
    t = d[0:3]
    qw = float(torch.clamp(d[6].abs(), max=1.0))
    s = float(d[7])
    return {
        "rotation_deg": math.degrees(2.0 * math.acos(qw)),
        "translation_norm": float(torch.linalg.norm(t)),
        "translation_vec": [float(x) for x in t],
        "scale": s,
    }
