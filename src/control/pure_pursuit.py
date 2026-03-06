from __future__ import annotations
import numpy as np
from .base import ControllerBase, ControlAction

class PurePursuitController(ControllerBase):
    name = "pure_pursuit"
    def __init__(self, track, target_speed: float = 12.0, lookahead: float = 5.0, kp_speed: float = 1.6, wheelbase: float = 2.7):
        self.track = track
        self.target_speed = target_speed
        self.lookahead = lookahead
        self.kp_speed = kp_speed
        self.wheelbase = wheelbase

    def control(self, obs: dict, dt: float) -> ControlAction:
        idx = obs["idx"]
        pos = obs["pos"]
        yaw = obs["yaw"]
        v = obs["v"]
        steps_ahead = max(1, int(self.lookahead / max(self.track.ds, 1e-6)))
        tidx = (idx + steps_ahead) % len(self.track.center)
        target = self.track.center[tidx]
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        alpha = np.arctan2(dy, dx) - yaw
        alpha = (alpha + np.pi) % (2*np.pi) - np.pi
        Ld = max(np.hypot(dx, dy), 1e-3)
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), Ld)
        accel = self.kp_speed * (self.target_speed - v)
        return ControlAction(float(steer), float(accel))
