from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .track import Track
from .vehicle import VehicleState, VehicleParams, step_kinematic

class TrackRLEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, dt: float = 0.05, target_speed: float = 12.0, timeout_s: float = 60.0, seed: int = 42):
        super().__init__()
        self.dt = dt
        self.target_speed = target_speed
        self.timeout_s = timeout_s
        self.np_random = np.random.default_rng(seed)
        self.track = Track.procedural(width=6.0)
        self.vp = VehicleParams()
        self.action_space = spaces.Box(low=np.array([-1.0,-1.0],dtype=np.float32),
                                       high=np.array([1.0,1.0],dtype=np.float32), dtype=np.float32)
        obs_dim = 4 + 8  # speed, ey, epsi, progress ratio + curvature preview
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.max_steps = int(timeout_s / dt)
        self.prev_s = 0.0
        self.prev_ey = 0.0
        self.prev_epsi = 0.0
        self.lap_complete = False

    def _get_obs(self):
        idx, s, ey, ref_yaw = self.track.project(np.array([self.state.x, self.state.y]))
        epsi = (self.state.yaw - ref_yaw + np.pi) % (2*np.pi) - np.pi
        curv = self.track.preview_curvature(idx)
        obs = np.concatenate([[self.state.v, ey, epsi, s/self.track.length], curv]).astype(np.float32)
        info = {"idx": idx, "s": s, "ey": ey, "epsi": epsi, "ref_yaw": ref_yaw}
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p0 = self.track.center[0]
        yaw0 = self.track.yaw[0]
        self.state = VehicleState(float(p0[0]), float(p0[1]), float(yaw0), 0.0)
        self.steps = 0
        self.prev_s = 0.0
        self.prev_ey = 0.0
        self.prev_epsi = 0.0
        self.lap_complete = False
        obs, info = self._get_obs()
        return obs, info

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0)) * self.vp.max_steer
        accel_cmd = float(np.clip(action[1], -1.0, 1.0))
        accel = accel_cmd * (self.vp.max_accel if accel_cmd >= 0 else self.vp.max_decel)
        self.state = step_kinematic(self.state, steer, accel, self.dt, self.vp)
        self.steps += 1

        obs, info = self._get_obs()
        s = info["s"]
        ds = self.track.progress_delta(self.prev_s, s)
        ey = info["ey"]
        epsi = info["epsi"]

        offtrack = abs(ey) > (self.track.width/2.0)
        timeout = self.steps >= self.max_steps
        lap_complete = self.track.crossed_finish(self.prev_s, s) and ds > 0.0 and self.steps > int(0.3*self.max_steps*0.2)
        self.lap_complete = lap_complete

        reward = 8.0 * max(ds, 0.0)
        reward -= 0.03                              # step penalty
        reward -= 0.02 * abs(steer)
        reward -= 0.01 * abs(accel)
        reward -= 0.10 * abs(epsi)
        reward -= 0.01 * abs(ey)
        if self.state.v < 0.4:
            reward -= 0.08
        if ds < -1e-3:
            reward -= 1.0
        if offtrack:
            reward -= 40.0
        if lap_complete:
            reward += 120.0

        terminated = bool(offtrack or lap_complete)
        truncated = bool((not terminated) and timeout)

        lap_time = self.steps * self.dt if lap_complete else np.nan
        info.update({
            "success": bool(lap_complete),
            "progress": float(max(s, 0.0)),
            "lap_time": float(lap_time if not np.isnan(lap_time) else self.steps*self.dt),
            "mean_speed": float(self.state.v),
        })

        self.prev_s = s
        self.prev_ey = ey
        self.prev_epsi = epsi
        return obs, float(reward), terminated, truncated, info
