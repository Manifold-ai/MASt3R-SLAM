"""Manifest schema + loader for the novel-pos pipeline (ADR-001 修订3).

Backend-free: stdlib + yaml only. Safe to import without a GPU or the compiled
`mast3r_slam_backends` extension.

The manifest describes a single-shot run: an existing GS (`.ply`); the novel-view render
image + AI completion image; the render camera pose and intrinsics (used to project the GS
into the render view); and the output `.ply`. CAMERA CONVENTION (contract §2): pose is
camera-to-world, quaternion **wxyz**, OpenCV axes (z forward); intrinsics are pixel
intrinsics for the render image's original resolution.
"""
import dataclasses
import pathlib
from typing import List, Optional

import yaml


@dataclasses.dataclass
class PoseSpec:
    translation: List[float]       # (3,) camera position in world
    quaternion_wxyz: List[float]   # (4,) rotation, wxyz order; camera-to-world


@dataclasses.dataclass
class IntrinsicsSpec:               # pixel intrinsics for render_image original resolution
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclasses.dataclass
class GateThresholds:
    max_rotation_deg: float = 5.0
    max_translation_norm: float = 0.1
    max_log_scale: float = 0.1


@dataclasses.dataclass
class Manifest:
    gs_ply: pathlib.Path
    render_image: pathlib.Path
    completion_image: pathlib.Path
    novel_pose: PoseSpec            # camera-to-world, wxyz, OpenCV; used for projection
    intrinsics: IntrinsicsSpec      # used for projection (-> K_frame)
    output_ply: pathlib.Path
    panorama_image: Optional[pathlib.Path] = None
    panorama_pose: Optional[PoseSpec] = None
    opacity_threshold: float = 0.5
    update_conf_percentile: float = 50.0
    gate: GateThresholds = dataclasses.field(default_factory=GateThresholds)


def _req(d: dict, key: str, ctx: str):
    if not isinstance(d, dict) or key not in d or d[key] is None:
        raise ValueError(f"manifest: missing required field '{ctx}{key}'")
    return d[key]


def _pose(d: dict, ctx: str) -> PoseSpec:
    t = _req(d, "translation", ctx)
    q = _req(d, "quaternion_wxyz", ctx)
    if len(t) != 3:
        raise ValueError(f"manifest: '{ctx}translation' must have 3 values, got {len(t)}")
    if len(q) != 4:
        raise ValueError(f"manifest: '{ctx}quaternion_wxyz' must have 4 values (wxyz), got {len(q)}")
    return PoseSpec(
        translation=[float(x) for x in t],
        quaternion_wxyz=[float(x) for x in q],
    )


def _intrinsics(d: dict, ctx: str) -> IntrinsicsSpec:
    for k in ("width", "height", "fx", "fy", "cx", "cy"):
        _req(d, k, ctx)
    return IntrinsicsSpec(
        width=int(d["width"]), height=int(d["height"]),
        fx=float(d["fx"]), fy=float(d["fy"]),
        cx=float(d["cx"]), cy=float(d["cy"]),
    )


def _resolve(base: pathlib.Path, p) -> pathlib.Path:
    path = pathlib.Path(str(p)).expanduser()
    return path if path.is_absolute() else (base / path)


def load_manifest(path: str) -> Manifest:
    mpath = pathlib.Path(path).expanduser().resolve()
    if not mpath.exists():
        raise FileNotFoundError(f"manifest not found: {mpath}")
    with open(mpath, "r") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"manifest: top level must be a mapping, got {type(d).__name__}")
    base = mpath.parent

    gs_ply = _resolve(base, _req(d, "gs_ply", ""))
    nv = _req(d, "novel_view", "")
    render_image = _resolve(base, _req(nv, "render_image", "novel_view."))
    completion_image = _resolve(base, _req(nv, "completion_image", "novel_view."))
    novel_pose = _pose(_req(nv, "pose", "novel_view."), "novel_view.pose.")
    intrinsics = _intrinsics(_req(nv, "intrinsics", "novel_view."), "novel_view.intrinsics.")
    output_ply = _resolve(base, _req(d, "output_ply", ""))

    pano = d.get("panorama") or {}
    panorama_image = _resolve(base, pano["image"]) if pano.get("image") else None
    panorama_pose = _pose(pano["pose"], "panorama.pose.") if pano.get("pose") else None

    opacity_threshold = float((d.get("gs_degrade") or {}).get("opacity_threshold", 0.5))
    update_conf_percentile = float((d.get("confidence") or {}).get("update_conf_percentile", 50.0))
    g = d.get("validation_gate") or {}
    gate = GateThresholds(
        max_rotation_deg=float(g.get("max_rotation_deg", 5.0)),
        max_translation_norm=float(g.get("max_translation_norm", 0.1)),
        max_log_scale=float(g.get("max_log_scale", 0.1)),
    )

    for label, p in [
        ("gs_ply", gs_ply),
        ("novel_view.render_image", render_image),
        ("novel_view.completion_image", completion_image),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"manifest: '{label}' file not found: {p}")

    output_ply.parent.mkdir(parents=True, exist_ok=True)

    return Manifest(
        gs_ply=gs_ply,
        render_image=render_image,
        completion_image=completion_image,
        novel_pose=novel_pose,
        intrinsics=intrinsics,
        output_ply=output_ply,
        panorama_image=panorama_image,
        panorama_pose=panorama_pose,
        opacity_threshold=opacity_threshold,
        update_conf_percentile=update_conf_percentile,
        gate=gate,
    )
