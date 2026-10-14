"""Manifest schema + loader for the novel-pos pipeline (ADR-001 修订2).

Backend-free: stdlib + yaml only. Safe to import without a GPU or the compiled
`mast3r_slam_backends` extension.

The manifest describes the fixed inputs of a single-shot run:
  - an existing Gaussian Splatting model (`.ply`),
  - the novel-view render image (rendered-from-GS RGB), its GS-WORLD per-pixel pointmap
    (renderer world-position pass) and a coverage mask, and the AI completion image,
  - optionally the render pose and reference intrinsics (sanity only; geometry comes from
    the render pointmap + the fitted similarity transform C),
  - the output `.ply` path, plus degrade / confidence / validation-gate thresholds.
"""
import dataclasses
import pathlib
from typing import List, Optional

import yaml


@dataclasses.dataclass
class PoseSpec:
    translation: List[float]       # (3,) world translation
    quaternion_xyzw: List[float]   # (4,) rotation, xyzw order
    scale: float = 1.0


@dataclasses.dataclass
class IntrinsicsSpec:               # parsed but unused in geometry (sanity only)
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
    render_pointmap: pathlib.Path       # GS-world per-pixel XYZ (.npy / .exr)
    render_coverage: pathlib.Path       # per-pixel valid mask (.npy / image)
    completion_image: pathlib.Path
    output_ply: pathlib.Path
    novel_pose: Optional[PoseSpec] = None            # sanity only
    reference_intrinsics: Optional[IntrinsicsSpec] = None  # sanity only
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
    q = _req(d, "quaternion_xyzw", ctx)
    if len(t) != 3:
        raise ValueError(f"manifest: '{ctx}translation' must have 3 values, got {len(t)}")
    if len(q) != 4:
        raise ValueError(f"manifest: '{ctx}quaternion_xyzw' must have 4 values (xyzw), got {len(q)}")
    return PoseSpec(
        translation=[float(x) for x in t],
        quaternion_xyzw=[float(x) for x in q],
        scale=float(d.get("scale", 1.0)),
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
    render_pointmap = _resolve(base, _req(nv, "render_pointmap", "novel_view."))
    render_coverage = _resolve(base, _req(nv, "render_coverage", "novel_view."))
    completion_image = _resolve(base, _req(nv, "completion_image", "novel_view."))
    output_ply = _resolve(base, _req(d, "output_ply", ""))

    # optional, sanity only
    nvp = nv.get("pose")
    novel_pose = _pose(nvp, "novel_view.pose.") if nvp else None
    ri = nv.get("reference_intrinsics")
    reference_intrinsics = None
    if ri:
        reference_intrinsics = IntrinsicsSpec(
            width=int(ri["width"]), height=int(ri["height"]),
            fx=float(ri["fx"]), fy=float(ri["fy"]),
            cx=float(ri["cx"]), cy=float(ri["cy"]),
        )

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

    # required input files must exist (fail fast with resolved path)
    for label, p in [
        ("gs_ply", gs_ply),
        ("novel_view.render_image", render_image),
        ("novel_view.render_pointmap", render_pointmap),
        ("novel_view.render_coverage", render_coverage),
        ("novel_view.completion_image", completion_image),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"manifest: '{label}' file not found: {p}")

    output_ply.parent.mkdir(parents=True, exist_ok=True)

    return Manifest(
        gs_ply=gs_ply,
        render_image=render_image,
        render_pointmap=render_pointmap,
        render_coverage=render_coverage,
        completion_image=completion_image,
        output_ply=output_ply,
        novel_pose=novel_pose,
        reference_intrinsics=reference_intrinsics,
        panorama_image=panorama_image,
        panorama_pose=panorama_pose,
        opacity_threshold=opacity_threshold,
        update_conf_percentile=update_conf_percentile,
        gate=gate,
    )
