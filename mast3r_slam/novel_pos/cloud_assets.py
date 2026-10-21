"""Sealed keyframe cloud assets for ADR-002 A (SLAM-C-003 §四). Backend-free: numpy only.

An asset is written ONCE (``assets/<id>.npz``) and never modified. The sealed part -- local
points X (ray-constrained, camera frame), original RGB, confidence, valid mask and grid shape;
point id = pixel index -- carries a sha256 ``content_hash`` that is re-checked before the
merge. The asset also records the solver inputs derived in the same inference (P0 anchors, K,
known camera, visibility record) and an input ``fingerprint``; an existing asset is reused only
when the fingerprint matches, otherwise the run stops instead of silently mixing inputs.
"""
import hashlib
import json
import os
import tempfile

import numpy as np

SEALED = ("X", "rgb", "conf", "valid", "shape")
# solver inputs derived in the same inference; hashed separately so reuse can verify them too
AUX = ("anchor_pix", "anchor_target", "anchor_q", "K", "cam_R", "cam_t", "vis_alpha",
       "render_coverage", "match_valid", "match_idx")


def array_digest(h, a):
    a = np.ascontiguousarray(a)
    h.update(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())


def arrays_hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        array_digest(h, np.asarray(a))
    return h.hexdigest()


def content_hash(asset: dict) -> str:
    return arrays_hash(*(asset[k] for k in SEALED))


def aux_hash(asset: dict) -> str:
    return arrays_hash(*(asset[k] for k in AUX if k in asset))


def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(parts: dict) -> str:
    """Stable hash of the inputs an asset was derived from (file hashes, pose, K, ...)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def asset_path(assets_dir, kf_id) -> str:
    return os.path.join(assets_dir, f"{kf_id}.npz")


def seal(path, asset: dict, meta: dict) -> str:
    """Write a new asset exactly once (atomic, refuses to overwrite). Returns its content hash."""
    digest = content_hash(asset)
    meta = dict(meta, content_hash=digest, aux_hash=aux_hash(asset))
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".npz.tmp")
    try:
        um = os.umask(0)
        os.umask(um)
        os.fchmod(fd, 0o666 & ~um)            # mkstemp makes 0600; keep assets readable like peers
        with os.fdopen(fd, "wb") as f:
            np.savez(f, meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
                     **{k: np.asarray(v) for k, v in asset.items()})
        try:
            os.link(tmp, path)                 # atomic publish; FileExistsError if already sealed
        except FileExistsError:
            raise
        except OSError:                        # no hard links here: exclusive create, then replace
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            os.replace(tmp, path)
    except FileExistsError:
        raise FileExistsError(f"sealed asset already exists: {path}") from None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return digest


def load(path) -> tuple:
    """Load an asset and verify its content hash. Returns (asset dict, meta dict)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        asset = {k: z[k] for k in z.files if k != "meta_json"}
    got = content_hash(asset)
    if got != meta.get("content_hash"):
        raise RuntimeError(f"sealed asset {path} was modified: hash {got} != {meta.get('content_hash')}")
    if aux_hash(asset) != meta.get("aux_hash"):
        raise RuntimeError(f"sealed asset {path}: anchors/camera fields were modified")
    return asset, meta


def load_or_none(path, expect_fingerprint):
    """Reuse an existing sealed asset when its inputs match; error when they do not."""
    if not os.path.exists(path):
        return None
    asset, meta = load(path)
    if meta.get("fingerprint") != expect_fingerprint:
        raise RuntimeError(f"sealed asset {path} was built from different inputs "
                           f"(fingerprint mismatch); use a new output_dir or delete it to rebuild")
    return asset, meta
