"""Frame construction + minimal keyframe stub for the standalone pose solve (ADR-001).

WARNING: importing this module pulls in mast3r_slam.frame -> mast3r_slam.mast3r_utils
-> mast3r_slam.matching -> `import mast3r_slam_backends` (the compiled CUDA extension).
Only import on a GPU machine with the backend built.
"""
import cv2
import numpy as np

from mast3r_slam.frame import create_frame, Frame


class KeyframeStub:
    """Minimal stand-in for SharedKeyframes that FrameTracker needs (single frame).

    FrameTracker only calls last_keyframe()/__len__/__getitem__ (and, inside track(),
    __setitem__, which the pose experiment never triggers).
    """

    def __init__(self, kf: Frame):
        self._kf = kf

    def last_keyframe(self) -> Frame:
        return self._kf

    def __len__(self) -> int:
        return 1

    def __getitem__(self, i) -> Frame:
        return self._kf

    def __setitem__(self, i, v) -> None:
        self._kf = v


def load_rgb01(path) -> np.ndarray:
    """Load an image as 0-1 float32 RGB numpy (matches resize_img / dataloader convention)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def build_frame(idx: int, rgb01: np.ndarray, T_WC, img_size: int = 512, device: str = "cuda:0") -> Frame:
    """Build a MASt3R-SLAM Frame from a 0-1 float RGB image and a lietorch.Sim3 pose."""
    return create_frame(idx, rgb01, T_WC, img_size=img_size, device=device)
