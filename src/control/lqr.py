from __future__ import annotations
import numpy as np
from scipy.linalg import solve_discrete_are
from .base import ControllerBase, ControlAction

class LQRController(ControllerBase):
    name = "lqr"
    def __init__(self, track, dt: float, target_speed: float = 12.0, kp_speed: float = 1.4, wheelbase: float = 2.7):
        self.track = track
        self.dt = dt
        self.target_speed = target_speed
        self.kp_speed = kp_speed
        self.wheelbase = wheelbase
        self._last_steer = 0.0
        self._last_idx = 0

    def _gain(self, vref: float):
        v = max(vref, 2.0)
        dt = self.dt
        A = np.array([
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, v, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ])
        B = np.array([[0.0], [0.0], [0.0], [v / self.wheelbase]])
        Q = np.diag([4.0, 0.4, 3.0, 0.3])
        R = np.array([[0.8]])
        P = solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K

    def control(self, obs: dict, dt: float) -> ControlAction:
        ey = obs["ey"]
        epsi = obs["epsi"]
        v = obs["v"]
        idx = obs["idx"]
        de = (ey - obs.get("prev_ey", ey)) / max(dt, 1e-6)
        dpsi = (epsi - obs.get("prev_epsi", epsi)) / max(dt, 1e-6)
        x = np.array([[ey], [de], [epsi], [dpsi]])
        K = self._gain(max(v, self.target_speed*0.8))
        ff = np.arctan(self.wheelbase * self.track.kappa[idx])
        steer = float(ff - (K @ x).item())
        accel = self.kp_speed * (self.target_speed - v)
        return ControlAction(steer=steer, accel=float(accel))
