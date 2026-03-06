from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Track:
    center: np.ndarray       # [N,2]
    yaw: np.ndarray          # [N]
    ds: float
    s: np.ndarray            # [N]
    length: float
    width: float
    left: np.ndarray         # [N,2]
    right: np.ndarray        # [N,2]
    kappa: np.ndarray        # [N]

    @staticmethod
    def procedural(num_points: int = 800, width: float = 6.0) -> "Track":
        th = np.linspace(0.0, 2*np.pi, num_points, endpoint=False)
        r = 26 + 6*np.sin(2*th + 0.4) + 4*np.sin(3*th - 0.6) + 2*np.cos(5*th + 0.5)
        x = r * np.cos(th) * 1.15
        y = r * np.sin(th) * 0.85
        center = np.column_stack([x, y])
        # close for differentiation
        nxt = np.roll(center, -1, axis=0)
        prv = np.roll(center, 1, axis=0)
        d = nxt - center
        seg = np.linalg.norm(d, axis=1)
        ds = float(np.mean(seg))
        yaw = np.arctan2(d[:, 1], d[:, 0])
        dyaw = np.unwrap(np.roll(yaw, -1) - yaw)
        kappa = dyaw / max(ds, 1e-6)
        normals = np.column_stack([-np.sin(yaw), np.cos(yaw)])
        left = center + normals * (width/2.0)
        right = center - normals * (width/2.0)
        s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        length = float(np.sum(seg))
        return Track(center=center, yaw=yaw, ds=ds, s=s, length=length, width=width, left=left, right=right, kappa=kappa)

    def project(self, pos: np.ndarray) -> tuple[int, float, float, float]:
        """Returns idx, progress s, lateral error ey, heading ref yaw."""
        diffs = self.center - pos[None, :]
        idx = int(np.argmin(np.sum(diffs**2, axis=1)))
        ref = self.center[idx]
        ref_yaw = self.yaw[idx]
        n = np.array([-np.sin(ref_yaw), np.cos(ref_yaw)])
        ey = float(np.dot(pos - ref, n))
        s = float(self.s[idx])
        return idx, s, ey, float(ref_yaw)

    def progress_delta(self, s_prev: float, s_now: float) -> float:
        ds = s_now - s_prev
        if ds < -0.5 * self.length:
            ds += self.length
        elif ds > 0.5 * self.length:
            ds -= self.length
        return ds

    def crossed_finish(self, s_prev: float, s_now: float) -> bool:
        # detect wrap from high s to low s while moving forward
        return s_prev > 0.85 * self.length and s_now < 0.15 * self.length

    def preview_curvature(self, idx: int, horizon: int = 8, stride: int = 8) -> np.ndarray:
        vals = []
        n = len(self.kappa)
        for i in range(horizon):
            vals.append(self.kappa[(idx + i*stride) % n])
        return np.array(vals, dtype=np.float32)
