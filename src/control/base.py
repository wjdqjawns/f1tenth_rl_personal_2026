from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ControlAction:
    steer: float
    accel: float

class ControllerBase:
    name = "base"
    def reset(self): ...
    def control(self, obs: dict, dt: float) -> ControlAction:
        raise NotImplementedError
