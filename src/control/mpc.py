from __future__ import annotations
import numpy as np
from .base import ControllerBase, ControlAction
from ..sim.vehicle import VehicleState, VehicleParams, step_kinematic

class MPCController(ControllerBase):
    name = "mpc"
    def __init__(self, track, dt: float, vehicle_params: VehicleParams, target_speed: float = 12.0,
                 horizon: int = 10, steer_candidates = (-0.35, -0.2, -0.08, 0.0, 0.08, 0.2, 0.35)):
        self.track = track
        self.dt = dt
        self.vp = vehicle_params
        self.target_speed = target_speed
        self.horizon = horizon
        self.steer_candidates = np.array(steer_candidates, dtype=float)

    def _cost(self, state: VehicleState, steer: float):
        total = 0.0
        st = VehicleState(state.x, state.y, state.yaw, state.v)
        accel = np.clip(1.2*(self.target_speed - st.v), -self.vp.max_decel, self.vp.max_accel)
        for _ in range(self.horizon):
            st = step_kinematic(st, steer, accel, self.dt, self.vp)
            idx, s, ey, ref_yaw = self.track.project(np.array([st.x, st.y]))
            epsi = (st.yaw - ref_yaw + np.pi) % (2*np.pi) - np.pi
            total += 4.0*ey*ey + 2.0*epsi*epsi + 0.15*(st.v-self.target_speed)**2 + 0.08*steer*steer
            if abs(ey) > self.track.width/2:
                total += 200.0
        return total

    def control(self, obs: dict, dt: float) -> ControlAction:
        state = VehicleState(float(obs["pos"][0]), float(obs["pos"][1]), float(obs["yaw"]), float(obs["v"]))
        costs = [self._cost(state, u) for u in self.steer_candidates]
        best = float(self.steer_candidates[int(np.argmin(costs))])
        accel = 1.3 * (self.target_speed - state.v)
        return ControlAction(steer=best, accel=float(accel))
