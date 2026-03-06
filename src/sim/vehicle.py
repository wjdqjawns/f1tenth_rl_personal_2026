from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float

@dataclass
class VehicleParams:
    wheelbase: float = 2.7
    max_steer: float = np.deg2rad(30.0)
    max_accel: float = 3.0
    max_decel: float = 6.0
    max_speed: float = 18.0
    min_speed: float = 0.0

def step_kinematic(state: VehicleState, steer: float, accel: float, dt: float, p: VehicleParams) -> VehicleState:
    steer = float(np.clip(steer, -p.max_steer, p.max_steer))
    accel = float(np.clip(accel, -p.max_decel, p.max_accel))
    v = float(np.clip(state.v + accel * dt, p.min_speed, p.max_speed))
    beta = np.arctan(0.5 * np.tan(steer))
    x = state.x + v * np.cos(state.yaw + beta) * dt
    y = state.y + v * np.sin(state.yaw + beta) * dt
    yaw = state.yaw + (v / max(p.wheelbase, 1e-6)) * np.tan(steer) * dt
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi
    return VehicleState(x=x, y=y, yaw=yaw, v=v)
