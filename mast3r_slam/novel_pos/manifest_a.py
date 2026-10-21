"""Multi-keyframe manifest for the ADR-002 A merge (SLAM-C-003 §三). Backend-free: yaml only.

Camera convention and intrinsics follow C-001 §二/§三 (camera-to-world, quaternion wxyz,
OpenCV axes; pixel intrinsics at the image's original resolution). Paths resolve relative to
the manifest. Example: config/manifest_a.example.yaml.
"""
import dataclasses
import pathlib
from typing import List, Optional

import yaml

from mast3r_slam.novel_pos.manifest import (
    IntrinsicsSpec, PoseSpec, _intrinsics, _pose, _req, _resolve,
)


@dataclasses.dataclass
class KeyframeSpec:
    id: str
    render_image: pathlib.Path
    completion_image: pathlib.Path
    pose: PoseSpec
    intrinsics: IntrinsicsSpec
    visibility_map: Optional[pathlib.Path] = None


@dataclasses.dataclass
class FixedViewSpec:
    id: str
    image: pathlib.Path
    pose: PoseSpec
    intrinsics: IntrinsicsSpec


@dataclasses.dataclass
class ManifestA:
    gs_ply: pathlib.Path
    keyframes: List[KeyframeSpec]
    output_dir: pathlib.Path
    fixed_views: List[FixedViewSpec] = dataclasses.field(default_factory=list)
    opacity_threshold: float = 0.5
    geo: dict = dataclasses.field(default_factory=dict)


def _id(d, ctx, seen):
    kid = str(_req(d, "id", ctx))
    if not kid or "/" in kid or kid in (".", ".."):
        raise ValueError(f"manifest: '{ctx}id' must be a plain non-empty name, got {kid!r}")
    if kid in seen:
        raise ValueError(f"manifest: duplicate id {kid!r}")
    seen.add(kid)
    return kid


def load_manifest_a(path: str) -> ManifestA:
    mpath = pathlib.Path(path).expanduser().resolve()
    if not mpath.exists():
        raise FileNotFoundError(f"manifest not found: {mpath}")
    with open(mpath, "r") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"manifest: top level must be a mapping, got {type(d).__name__}")
    base = mpath.parent
    gs_ply = _resolve(base, _req(d, "gs_ply", ""))
    kfs_raw = _req(d, "keyframes", "")
    if not isinstance(kfs_raw, list) or not kfs_raw:
        raise ValueError("manifest: 'keyframes' must be a non-empty list")
    seen, files = set(), [("gs_ply", gs_ply)]
    keyframes = []
    for n, k in enumerate(kfs_raw):
        ctx = f"keyframes[{n}]."
        kid = _id(k, ctx, seen)
        vm = k.get("visibility_map")
        kf = KeyframeSpec(
            id=kid,
            render_image=_resolve(base, _req(k, "render_image", ctx)),
            completion_image=_resolve(base, _req(k, "completion_image", ctx)),
            pose=_pose(_req(k, "pose", ctx), ctx + "pose."),
            intrinsics=_intrinsics(_req(k, "intrinsics", ctx), ctx + "intrinsics."),
            visibility_map=_resolve(base, vm) if vm is not None else None,
        )
        files += [(ctx + "render_image", kf.render_image),
                  (ctx + "completion_image", kf.completion_image)]
        if kf.visibility_map is not None:
            files.append((ctx + "visibility_map", kf.visibility_map))
        keyframes.append(kf)
    fixed_views = []
    for n, v in enumerate(d.get("fixed_views") or []):
        ctx = f"fixed_views[{n}]."
        fv = FixedViewSpec(
            id=_id(v, ctx, seen),
            image=_resolve(base, _req(v, "image", ctx)),
            pose=_pose(_req(v, "pose", ctx), ctx + "pose."),
            intrinsics=_intrinsics(_req(v, "intrinsics", ctx), ctx + "intrinsics."),
        )
        files.append((ctx + "image", fv.image))
        fixed_views.append(fv)
    for label, p in files:
        if not p.exists():
            raise FileNotFoundError(f"manifest: '{label}' file not found: {p}")
    geo = d.get("geo") or {}
    if not isinstance(geo, dict):
        raise ValueError("manifest: 'geo' must be a mapping of parameter overrides")
    return ManifestA(
        gs_ply=gs_ply,
        keyframes=keyframes,
        output_dir=_resolve(base, _req(d, "output_dir", "")),
        fixed_views=fixed_views,
        opacity_threshold=float((d.get("gs_degrade") or {}).get("opacity_threshold", 0.5)),
        geo=geo,
    )
