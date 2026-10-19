"""Frame construction helpers for the novel-pos pipeline (ADR-001).

WARNING: importing this module pulls in mast3r_slam.frame -> mast3r_slam.mast3r_utils
-> mast3r_slam.matching -> `import mast3r_slam_backends` (the compiled CUDA extension).
Only import on a GPU machine with the backend built.
"""
import cv2
import numpy as np

from mast3r_slam.frame import create_frame, Frame
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.novel_pos import visibility


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


def load_visibility_alpha(path, render_hw, img_size: int = 512):
    """Load the render visibility map (C-002) and align it to the MASt3R grid.

    The map has the render image's size, so one ``resize_img`` call gives it the same scale,
    interpolation and center crop as the render. Returns (alpha (H,W) float32 in [0,1],
    true_shape (H,W)).
    """
    # no color conversion; <8-bit gray PNGs are expanded to 8-bit. Palette PNGs decode to
    # BGR(A): accepted only with a pure-gray palette and no tRNS, otherwise rejected below.
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not read visibility_map: {path}")
    gray = visibility.as_gray_u8(raw, render_hw)
    a01 = np.repeat(gray[..., None], 3, axis=2).astype(np.float32) / 255.0  # ImgNorm needs 3 ch
    res = resize_img(a01, img_size)
    alpha = res["unnormalized_img"][..., 0].astype(np.float32) / 255.0
    return alpha, tuple(int(x) for x in res["true_shape"][0])
