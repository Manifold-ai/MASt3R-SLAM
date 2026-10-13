"""Manifest schema + loader for the novel-pos pipeline (ADR-001).

Backend-free: stdlib + yaml only. Safe to import without a GPU or the compiled
`mast3r_slam_backends` extension.

The manifest describes the fixed inputs of a single-shot run:
  - an existing Gaussian Splatting model (`.ply`),
  - a panorama image + its pose (panorama camera center == GS world origin; v1 metadata only),
  - novel-view reference intrinsics (parsed but UNUSED in v1),
  - the novel-view render image (rendered-from-GS original) and AI completion image,
  - the novel-view camera pose (the render/operating pose; v1 uses it directly),
  - the output `.ply` path.
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
class IntrinsicsSpec:               # parsed but unused in v1 (use_calib forced off)
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclasses.dataclass
class Manifest:
    gs_ply: pathlib.Path
    render_image: pathlib.Path
    completion_image: pathlib.Path
    novel_pose: PoseSpec                      # operating/render pose (world <- camera)
    output_ply: pathlib.Path
    panorama_image: Optional[pathlib.Path] = None
    panorama_pose: Optional[PoseSpec] = None
    reference_intrinsics: Optional[IntrinsicsSpec] = None
    opacity_threshold: float = 0.5
    update_conf_percentile: float = 50.0


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
    completion_image = _resolve(base, _req(nv, "completion_image", "novel_view."))
    novel_pose = _pose(_req(nv, "pose", "novel_view."), "novel_view.pose.")
    output_ply = _resolve(base, _req(d, "output_ply", ""))

    # panorama: optional, metadata only in v1
    pano = d.get("panorama") or {}
    panorama_image = _resolve(base, pano["image"]) if pano.get("image") else None
    panorama_pose = _pose(pano["pose"], "panorama.pose.") if pano.get("pose") else None

    # reference intrinsics: optional, parsed but unused in v1
    ri = nv.get("reference_intrinsics")
    reference_intrinsics = None
    if ri:
        reference_intrinsics = IntrinsicsSpec(
            width=int(ri["width"]), height=int(ri["height"]),
            fx=float(ri["fx"]), fy=float(ri["fy"]),
            cx=float(ri["cx"]), cy=float(ri["cy"]),
        )

    opacity_threshold = float((d.get("gs_degrade") or {}).get("opacity_threshold", 0.5))
    update_conf_percentile = float((d.get("confidence") or {}).get("update_conf_percentile", 50.0))

    # required input files must exist (fail fast with resolved path)
    for label, p in [
        ("gs_ply", gs_ply),
        ("novel_view.render_image", render_image),
        ("novel_view.completion_image", completion_image),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"manifest: '{label}' file not found: {p}")

    # make sure the output directory is writable
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    return Manifest(
        gs_ply=gs_ply,
        render_image=render_image,
        completion_image=completion_image,
        novel_pose=novel_pose,
        output_ply=output_ply,
        panorama_image=panorama_image,
        panorama_pose=panorama_pose,
        reference_intrinsics=reference_intrinsics,
        opacity_threshold=opacity_threshold,
        update_conf_percentile=update_conf_percentile,
    )
