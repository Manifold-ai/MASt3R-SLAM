"""Sim(3) helpers for the ADR-002 A geometric merge (SLAM-C-003). Backend-free: numpy only.

A transform ``T = (s, R, t)`` maps local points to world: ``x_w = s * R @ x + t`` with
``s > 0`` and ``det(R) = +1`` (no reflection, unlike the ADR-001 C fit).

Perturbations follow the MASt3R-SLAM backend: ``xi = (tau, phi, sigma)`` (translation,
rotation, log-scale) applied on the LEFT, ``T <- delta(xi) * T``. The point Jacobian of
``delta(xi) * w`` at ``xi = 0`` is ``[I, -[w]x, w]`` (same as ``geometry.act_Sim3``). We use
the retraction ``delta(xi) w = e^sigma Exp(phi) w + tau``; it matches the Sim(3) exponential
to first order, which is all Gauss-Newton needs.
"""
import dataclasses

import numpy as np


def skew(v) -> np.ndarray:
    """(...,3) -> (...,3,3) cross-product matrices, ``skew(a) @ b == a x b``."""
    v = np.asarray(v, dtype=np.float64)
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def so3_exp(phi) -> np.ndarray:
    """Rotation vector -> rotation matrix (Rodrigues)."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(phi))
    K = skew(phi)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def so3_log(R) -> np.ndarray:
    """Rotation matrix -> rotation vector, angle in [0, pi]."""
    R = np.asarray(R, dtype=np.float64)
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * w
    if np.pi - th < 1e-6:                      # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        return th * axis / np.linalg.norm(axis)
    return th / (2 * np.sin(th)) * w


def so3_left_jacobian_inv(phi) -> np.ndarray:
    """J_l^-1(phi): d log(Exp(dphi) Exp(phi)) / d dphi at dphi = 0."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(phi))
    K = skew(phi)
    if th < 1e-6:
        return np.eye(3) - 0.5 * K + K @ K / 12.0
    c = 1.0 / th**2 - (1 + np.cos(th)) / (2 * th * np.sin(th))
    return np.eye(3) - 0.5 * K + c * K @ K


def rotation_angle_deg(R) -> float:
    return float(np.degrees(np.linalg.norm(so3_log(R))))


def matrix_to_quat_wxyz(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion [w, x, y, z] with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def point_jacobian(W) -> np.ndarray:
    """(N,3) world points -> (N,3,7) d(delta(xi) w)/d xi at xi = 0: ``[I, -[w]x, w]``."""
    W = np.asarray(W, dtype=np.float64)
    J = np.zeros((W.shape[0], 3, 7))
    J[:, :, 0:3] = np.eye(3)
    J[:, :, 3:6] = -skew(W)
    J[:, :, 6] = W
    return J


@dataclasses.dataclass
class Sim3:
    s: float
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.s = float(self.s)
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @staticmethod
    def identity() -> "Sim3":
        return Sim3(1.0, np.eye(3), np.zeros(3))

    def act(self, X) -> np.ndarray:
        """local (N,3) -> world (N,3)."""
        return self.s * (np.asarray(X, dtype=np.float64) @ self.R.T) + self.t

    def inv_act(self, W) -> np.ndarray:
        """world (N,3) -> local (N,3)."""
        return ((np.asarray(W, dtype=np.float64) - self.t) @ self.R) / self.s

    def retract(self, xi) -> "Sim3":
        """Left update ``delta(xi) * self`` with ``delta(xi) w = e^sigma Exp(phi) w + tau``."""
        xi = np.asarray(xi, dtype=np.float64).reshape(7)
        dR, ds = so3_exp(xi[3:6]), float(np.exp(xi[6]))
        return Sim3(ds * self.s, dR @ self.R, ds * (dR @ self.t) + xi[0:3])

    def to_dict(self) -> dict:
        return {"scale": self.s, "quaternion_wxyz": matrix_to_quat_wxyz(self.R).tolist(),
                "translation": self.t.tolist()}
