"""Gaussian-Splatting .ply reader, degradation to point cloud, and PLY writer (ADR-001).

Backend-free: numpy + plyfile only. Deliberately does NOT import mast3r_slam.evaluate
(which transitively imports the compiled CUDA backend); `write_ply` below mirrors
`mast3r_slam.evaluate.save_ply` (same vertex schema, binary PLY).
"""
import numpy as np
from plyfile import PlyData, PlyElement

# SH band-0 constant (3DGS / INRIA convention). The ADR rounds this to 0.282.
SH_C0 = 0.28209479177387814


def read_gaussian_ply(path) -> dict:
    """Tolerant 3DGS `.ply` reader.

    Reads by field name. Requires x/y/z; `f_dc_*` (SH DC color) and `opacity` (logit)
    are optional with safe fallbacks. Higher-order SH (`f_rest_*`), `scale_*`, `rot_*`,
    and normals (`nx/ny/nz`, e.g. from KleinStudio) are ignored in v1.
    """
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    names = set(v.dtype.names)
    for f in ("x", "y", "z"):
        if f not in names:
            raise ValueError(f"GS ply missing required field '{f}'; found {sorted(names)}")
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    if all(f in names for f in ("f_dc_0", "f_dc_1", "f_dc_2")):
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1).astype(np.float32)
    else:
        print("[novel_pos] WARNING: GS ply has no f_dc_* fields; using gray color.")
        f_dc = np.zeros_like(xyz)

    opacity_logit = v["opacity"].astype(np.float32) if "opacity" in names else None
    if opacity_logit is None:
        print("[novel_pos] WARNING: GS ply has no 'opacity' field; keeping all points.")

    return {"xyz": xyz, "f_dc": f_dc, "opacity_logit": opacity_logit}


def degrade_gaussians_to_points(gs: dict, opacity_threshold: float = 0.5):
    """Gaussian degradation: centers -> points, SH0 -> color, sigmoid(opacity) filter.

    Returns (points (M,3) float32, colors (M,3) uint8).
    """
    xyz = gs["xyz"]
    color = np.clip(0.5 + SH_C0 * gs["f_dc"], 0.0, 1.0)
    colors_u8 = (color * 255.0).round().astype(np.uint8)

    opacity_logit = gs["opacity_logit"]
    if opacity_logit is not None:
        alpha = 1.0 / (1.0 + np.exp(-opacity_logit))  # sigmoid
        keep = alpha >= opacity_threshold
        xyz = xyz[keep]
        colors_u8 = colors_u8[keep]
    return xyz.astype(np.float32), colors_u8


def write_ply(path, points: np.ndarray, colors: np.ndarray):
    """Binary PLY with x,y,z (f4) + red,green,blue (u1). Mirrors evaluate.save_ply."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors).astype(np.uint8).reshape(-1, 3)
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points[:, 0], points[:, 1], points[:, 2]
    pcd["red"], pcd["green"], pcd["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    element = PlyElement.describe(pcd, "vertex")
    PlyData([element], text=False).write(str(path))
