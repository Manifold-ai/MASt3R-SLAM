"""Pose helpers for novel-pos (ADR-001). Backend-free: torch + lietorch only.

Only `decompose_relative_sim3` remains — used by the report-only validation gate to read a
relative Sim3's deviation from identity. (Quaternion->rotation for the GS projection lives
in `gs_project`, pure numpy, since the manifest pose is consumed there, not via lietorch.)
"""
import math

import torch  # noqa: F401  (kept for parity with the repo's lietorch/torch usage)


def decompose_relative_sim3(T_rel) -> dict:
    """Deviation-from-identity of a relative Sim3: rotation(deg) / translation / scale.

    Reads .data directly: ``[tx, ty, tz, qx, qy, qz, qw, s]`` (lietorch Sim3 layout).
    rotation angle = ``2*acos(|qw|)`` (abs handles the quaternion double cover).
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
