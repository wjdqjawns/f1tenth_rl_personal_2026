import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class VehicleParams:
    wheelbase: float = 2.8
    max_steer: float = np.deg2rad(28.0)
    max_accel: float = 7.0
    max_brake: float = 10.0
    max_speed: float = 55.0
    min_speed: float = 0.0
    drag_coeff: float = 0.0035
    rolling_resist: float = 0.08
    yaw_rate_limit: float = 1.6


class F1TrackEnv(gym.Env):
    """
    Lap-time oriented closed-track racing environment.

    The centerline is only used as a geometric reference for progress and
    heading computation. The reward does not try to keep the agent on the
    centerline; instead it rewards forward progress and penalizes time waste,
    reverse motion, excessive oscillation, and going off-track.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(self, render_mode: Optional[str] = None, dt: float = 0.05, lap_target: int = 1):
        super().__init__()
        self.render_mode = render_mode
        self.dt = dt
        self.lap_target = lap_target
        self.vehicle = VehicleParams()

        self.track_center, self.track_width = self._build_track(num_points=900)
        self.track_s = self._arc_length(self.track_center)
        self.track_length = float(self.track_s[-1] + np.linalg.norm(self.track_center[0] - self.track_center[-1]))
        self.track_curvature = self._estimate_curvature(self.track_center)
        self.track_tangent, self.track_normal = self._build_track_frames(self.track_center)
        self.left_boundary = self.track_center + self.track_normal * (self.track_width / 2.0)
        self.right_boundary = self.track_center - self.track_normal * (self.track_width / 2.0)
        self.start_line_a = self.left_boundary[0].copy()
        self.start_line_b = self.right_boundary[0].copy()
        self.start_forward = self.track_tangent[0].copy()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_high = np.array(
            [
                self.vehicle.max_speed,      # speed
                self.vehicle.yaw_rate_limit * 2.0,
                1.0,                         # lateral normalized
                np.pi,                       # heading error
                self.track_width,
                self.track_width,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0,                    # prev steer/throttle
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.max_steps = 5000
        self._screen = None
        self._clock = None
        self._font = None
        self.state = None
        self.prev_pos = None
        self.prev_s = 0.0
        self.steps = 0
        self.completed_laps = 0
        self.total_progress = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.just_crossed_start = False
        self.episode_history = {}

    @staticmethod
    def _build_track(num_points: int = 900) -> Tuple[np.ndarray, float]:
        theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        r = 34.0 + 7.0 * np.sin(3.0 * theta + 0.25) + 3.5 * np.sin(5.0 * theta - 0.8)
        x = 1.33 * r * np.cos(theta) + 5.5 * np.sin(2.0 * theta)
        y = 0.96 * r * np.sin(theta) + 6.5 * np.sin(theta) * np.cos(2.0 * theta)
        center = np.stack([x, y], axis=1).astype(np.float32)
        width = 12.0
        return center, width

    @staticmethod
    def _arc_length(points: np.ndarray) -> np.ndarray:
        diffs = np.roll(points, -1, axis=0) - points
        seg = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        return s.astype(np.float32)

    @staticmethod
    def _estimate_curvature(points: np.ndarray) -> np.ndarray:
        p_prev = np.roll(points, 1, axis=0)
        p = points
        p_next = np.roll(points, -1, axis=0)
        a = p - p_prev
        b = p_next - p
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(p_next - p_prev, axis=1) + 1e-6
        curvature = 2.0 * cross / denom
        return np.clip(curvature, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _build_track_frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        return tangent.astype(np.float32), normal.astype(np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _segment_intersection(p1, p2, q1, q2) -> bool:
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

    def _project_to_centerline(self, pos: np.ndarray):
        points = self.track_center
        next_points = np.roll(points, -1, axis=0)
        seg = next_points - points
        rel = pos[None, :] - points
        seg_len_sq = np.sum(seg * seg, axis=1) + 1e-8
        tau = np.clip(np.sum(rel * seg, axis=1) / seg_len_sq, 0.0, 1.0)
        proj = points + seg * tau[:, None]
        d = np.linalg.norm(pos[None, :] - proj, axis=1)
        idx = int(np.argmin(d))
        tangent = seg[idx] / (np.linalg.norm(seg[idx]) + 1e-8)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        lateral = float(np.dot(pos - proj[idx], normal))
        s = float(self.track_s[idx] + tau[idx] * np.linalg.norm(seg[idx]))
        heading_ref = math.atan2(float(tangent[1]), float(tangent[0]))
        return s, proj[idx], tangent.astype(np.float32), normal, lateral, heading_ref, idx

    def _lookahead_curvature(self, s_now: float, n_samples: int = 6, spacing_m: float = 12.0) -> np.ndarray:
        vals = []
        n = len(self.track_center)
        ds = self.track_length / n
        for k in range(1, n_samples + 1):
            s_target = (s_now + k * spacing_m) % self.track_length
            idx = int(s_target / ds) % n
            vals.append(self.track_curvature[idx])
        return np.array(vals, dtype=np.float32)

    def _make_obs(self) -> np.ndarray:
        x, y, yaw, v, yaw_rate = self.state
        pos = np.array([x, y], dtype=np.float32)
        s, _, _, _, lateral, heading_ref, _ = self._project_to_centerline(pos)
        heading_error = self._wrap_angle(yaw - heading_ref)
        half_w = self.track_width / 2.0
        dist_left = half_w - lateral
        dist_right = half_w + lateral
        preview = self._lookahead_curvature(s_now=s, n_samples=6, spacing_m=14.0)
        obs = np.array(
            [
                v,
                yaw_rate,
                np.clip(lateral / half_w, -1.5, 1.5),
                heading_error,
                dist_left,
                dist_right,
                *preview,
                self.prev_action[0],
                self.prev_action[1],
            ],
            dtype=np.float32,
        )
        return obs

    def _append_history(self, reward: float, progress_delta: float, accel_cmd: float, steer: float, throttle: float, info: Dict):
        x, y, yaw, v, yaw_rate = self.state
        self.episode_history["t"].append(self.steps * self.dt)
        self.episode_history["x"].append(x)
        self.episode_history["y"].append(y)
        self.episode_history["yaw"].append(yaw)
        self.episode_history["speed"].append(v)
        self.episode_history["yaw_rate"].append(yaw_rate)
        self.episode_history["steer"].append(steer)
        self.episode_history["steer_cmd"].append(float(self.prev_action[0]))
        self.episode_history["throttle_cmd"].append(float(self.prev_action[1]))
        self.episode_history["accel"].append(accel_cmd)
        self.episode_history["reward"].append(reward)
        self.episode_history["progress_delta"].append(progress_delta)
        self.episode_history["progress"].append(info["progress_m"])
        self.episode_history["lateral_error"].append(info["lateral_error_m"])
        self.episode_history["heading_error"].append(info["heading_error_rad"])

    def get_episode_history(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in self.episode_history.items()}

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        start_idx = 0 if options is None or "start_idx" not in options else int(options["start_idx"]) % len(self.track_center)
        p0 = self.track_center[start_idx]
        tangent = self.track_tangent[start_idx]
        yaw0 = math.atan2(float(tangent[1]), float(tangent[0]))
        start_speed = 8.0
        self.state = np.array([p0[0], p0[1], yaw0, start_speed, 0.0], dtype=np.float32)
        self.prev_pos = p0.copy()
        self.prev_s = float(self.track_s[start_idx])
        self.steps = 0
        self.completed_laps = 0
        self.total_progress = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.just_crossed_start = False
        self.episode_history = {k: [] for k in [
            "t", "x", "y", "yaw", "speed", "yaw_rate", "steer", "steer_cmd", "throttle_cmd",
            "accel", "reward", "progress_delta", "progress", "lateral_error", "heading_error"
        ]}
        return self._make_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))
        steer = steer_cmd * self.vehicle.max_steer

        x, y, yaw, v, _ = [float(s) for s in self.state]
        if throttle_cmd >= 0.0:
            accel_cmd = throttle_cmd * self.vehicle.max_accel
        else:
            accel_cmd = throttle_cmd * self.vehicle.max_brake

        accel_net = accel_cmd - self.vehicle.drag_coeff * v * v - self.vehicle.rolling_resist * np.sign(v)
        v = float(np.clip(v + accel_net * self.dt, self.vehicle.min_speed, self.vehicle.max_speed))
        yaw_rate = float(v / self.vehicle.wheelbase * math.tan(steer))
        x += v * math.cos(yaw) * self.dt
        y += v * math.sin(yaw) * self.dt
        yaw = self._wrap_angle(yaw + yaw_rate * self.dt)
        self.state = np.array([x, y, yaw, v, yaw_rate], dtype=np.float32)
        self.steps += 1

        pos = np.array([x, y], dtype=np.float32)
        s, _, tangent, _, lateral_error, heading_ref, _ = self._project_to_centerline(pos)
        heading_error = self._wrap_angle(yaw - heading_ref)
        progress_delta = s - self.prev_s
        if progress_delta < -0.5 * self.track_length:
            progress_delta += self.track_length
        elif progress_delta > 0.5 * self.track_length:
            progress_delta -= self.track_length

        reverse = progress_delta < -0.2
        forward_progress = max(progress_delta, 0.0)
        self.total_progress += forward_progress

        crossed_start = self._segment_intersection(self.prev_pos, pos, self.start_line_a, self.start_line_b)
        forward_crossing = float(np.dot(pos - self.prev_pos, self.start_forward)) > 0.0
        lap_valid = self.total_progress > 0.75 * self.track_length and self.steps > 50
        lap_increment = 1 if crossed_start and forward_crossing and lap_valid and not self.just_crossed_start else 0
        if lap_increment:
            self.completed_laps += 1
            self.just_crossed_start = True
        else:
            self.just_crossed_start = crossed_start and forward_crossing

        half_w = self.track_width / 2.0
        offtrack = abs(lateral_error) > half_w
        timeout = self.steps >= self.max_steps
        success = self.completed_laps >= self.lap_target
        stopped = v < 0.5 and self.steps > 80
        spinning = abs(yaw_rate) > self.vehicle.yaw_rate_limit and forward_progress < 0.05

        reward = 0.0
        reward += 6.0 * forward_progress
        reward -= 0.08                              # time penalty
        reward -= 0.25 * abs(steer_cmd - float(self.prev_action[0]))
        reward -= 0.06 * abs(throttle_cmd - float(self.prev_action[1]))
        reward -= 0.015 * abs(yaw_rate)
        if reverse:
            reward -= 3.0 + 2.0 * abs(progress_delta)
        if spinning:
            reward -= 2.0
        if stopped:
            reward -= 1.0
        if offtrack:
            reward -= 80.0
        if success:
            reward += 150.0

        terminated = bool(offtrack or success)
        truncated = bool(timeout)

        info = {
            "lateral_error_m": float(lateral_error),
            "heading_error_rad": float(heading_error),
            "speed_mps": float(v),
            "yaw_rate_rps": float(yaw_rate),
            "completed_laps": int(self.completed_laps),
            "lap_time_s": float(self.steps * self.dt),
            "progress_m": float(self.total_progress),
            "track_length_m": float(self.track_length),
            "success": bool(success),
            "offtrack": bool(offtrack),
            "reverse": bool(reverse),
            "spinning": bool(spinning),
        }

        self.prev_action = np.array([steer_cmd, throttle_cmd], dtype=np.float32)
        self._append_history(float(reward), float(progress_delta), float(accel_net), float(steer), float(throttle_cmd), info)
        self.prev_pos = pos.copy()
        self.prev_s = s

        return self._make_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        import pygame

        screen_w, screen_h = 1200, 800
        world = self.track_center
        min_xy = world.min(axis=0)
        max_xy = world.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        scale = min((screen_w - 120) / span[0], (screen_h - 120) / span[1])

        def to_screen(p):
            px = (p[0] - min_xy[0]) * scale + 60
            py = screen_h - ((p[1] - min_xy[1]) * scale + 60)
            return int(px), int(py)

        if self._screen is None:
            pygame.init()
            pygame.font.init()
            self._font = pygame.font.SysFont('Arial', 20)
            if self.render_mode == 'human':
                pygame.display.init()
                self._screen = pygame.display.set_mode((screen_w, screen_h))
            else:
                self._screen = pygame.Surface((screen_w, screen_h))
        if self._clock is None:
            self._clock = pygame.time.Clock()

        canvas = self._screen
        canvas.fill((22, 22, 22))

        pygame.draw.lines(canvas, (70, 70, 70), True, [to_screen(p) for p in self.left_boundary], 4)
        pygame.draw.lines(canvas, (70, 70, 70), True, [to_screen(p) for p in self.right_boundary], 4)
        pygame.draw.lines(canvas, (120, 160, 120), True, [to_screen(p) for p in self.track_center], 1)
        pygame.draw.line(canvas, (255, 255, 0), to_screen(self.start_line_a), to_screen(self.start_line_b), 4)

        history = self.get_episode_history()
        if len(history.get('x', [])) > 1:
            traj = np.stack([history['x'], history['y']], axis=1)
            pygame.draw.lines(canvas, (70, 170, 255), False, [to_screen(p) for p in traj], 2)

        x, y, yaw, v, _ = self.state
        car_len = 4.5
        car_w = 2.0
        corners = np.array([
            [ car_len / 2,  car_w / 2],
            [ car_len / 2, -car_w / 2],
            [-car_len / 2, -car_w / 2],
            [-car_len / 2,  car_w / 2],
        ], dtype=np.float32)
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]], dtype=np.float32)
        car_pts = corners @ rot.T + np.array([x, y], dtype=np.float32)
        pygame.draw.polygon(canvas, (220, 40, 40), [to_screen(p) for p in car_pts])

        text_lines = [
            f'speed: {v:5.2f} m/s',
            f'time: {self.steps * self.dt:5.2f} s',
            f'progress: {self.total_progress:6.1f} / {self.track_length:6.1f} m',
            f'laps: {self.completed_laps}',
            f'steer cmd: {self.prev_action[0]:+.2f}',
            f'throttle cmd: {self.prev_action[1]:+.2f}',
        ]
        for i, line in enumerate(text_lines):
            txt = self._font.render(line, True, (240, 240, 240))
            canvas.blit(txt, (20, 20 + 24 * i))

        if self.render_mode == 'human':
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None