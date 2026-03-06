from __future__ import annotations
import argparse, time, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from .utils.config import load_config
from .sim.track import Track
from .sim.vehicle import VehicleParams, VehicleState, step_kinematic
from .sim.logger import StepLogger, write_summary
from .analysis.common import save_series_and_trajectory
from .analysis.plot_rl_training import plot_csv
from .sim.env import TrackRLEnv
from .control.pure_pursuit import PurePursuitController
from .control.lqr import LQRController
from .control.mpc import MPCController

EXP_DATA = Path("experiments/data")
EXP_FIG = Path("experiments/fig")
EXP_LOG = Path("experiments/log")
MODELS = Path("models")
for d in [EXP_DATA, EXP_FIG, EXP_LOG, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

class RLMetricCallback(BaseCallback):
    def __init__(self, dt: float, save_csv_path: str):
        super().__init__()
        self.dt = float(dt)
        self.save_csv_path = Path(save_csv_path)
        self.rows = []
        self.episode = 0
        self.best_lap = np.inf

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue
            self.episode += 1
            ep_len = int(info["episode"]["l"])
            ep_rew = float(info["episode"]["r"])
            ep_t = ep_len * self.dt
            success = int(bool(info.get("success", False)))
            progress = float(info.get("progress", 0.0))
            lap_time = float(info.get("lap_time", ep_t))
            if success:
                self.best_lap = min(self.best_lap, lap_time)
            row = {
                "episode": self.episode,
                "reward": ep_rew,
                "episode_length": ep_len,
                "episode_time_s": ep_t,
                "success": success,
                "progress_m": progress,
                "lap_time_s": lap_time,
                "best_lap_s": self.best_lap if np.isfinite(self.best_lap) else ep_t,
                "mean_speed_mps": float(info.get("mean_speed", 0.0)),
            }
            self.rows.append(row)
        return True

    def _on_training_end(self) -> None:
        self.save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.save_csv_path, index=False)

def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def controller_from_cfg(cfg, track, vp):
    name = cfg["controller"]["name"]
    target_speed = float(cfg["controller"].get("target_speed_mps", cfg["sim"].get("target_speed_mps", 12.0)))
    dt = float(cfg["sim"]["dt"])
    if name == "pure_pursuit":
        return PurePursuitController(track, target_speed=target_speed,
                                     lookahead=float(cfg["controller"].get("lookahead_m", 5.0)),
                                     kp_speed=float(cfg["controller"].get("kp_speed", 1.6)),
                                     wheelbase=vp.wheelbase)
    if name == "lqr":
        return LQRController(track, dt=dt, target_speed=target_speed,
                             kp_speed=float(cfg["controller"].get("kp_speed", 1.4)),
                             wheelbase=vp.wheelbase)
    if name == "mpc":
        return MPCController(track, dt=dt, vehicle_params=vp,
                             target_speed=target_speed,
                             horizon=int(cfg["controller"].get("horizon", 10)))
    raise ValueError(f"Unsupported controller: {name}")

def run_tracking(cfg: dict):
    ctrl_name = cfg["controller"]["name"]
    dt = float(cfg["sim"]["dt"])
    timeout_s = float(cfg["sim"]["timeout_s"])
    max_steps = int(timeout_s / dt)
    track = Track.procedural(width=float(cfg["track"]["width_m"]))
    vp = VehicleParams(
        wheelbase=float(cfg["vehicle"]["wheelbase_m"]),
        max_steer=np.deg2rad(float(cfg["vehicle"]["max_steer_deg"])),
        max_accel=float(cfg["vehicle"]["max_accel_mps2"]),
        max_decel=float(cfg["vehicle"]["max_decel_mps2"]),
        max_speed=float(cfg["vehicle"]["max_speed_mps"]),
    )
    controller = controller_from_cfg(cfg, track, vp)

    p0 = track.center[0]
    st = VehicleState(float(p0[0]), float(p0[1]), float(track.yaw[0]), 0.0)
    logger = StepLogger()
    prev_s = 0.0
    prev_ey = 0.0
    prev_epsi = 0.0
    success = False
    compute_ms = []
    for k in range(max_steps):
        idx, s, ey, ref_yaw = track.project(np.array([st.x, st.y]))
        epsi = wrap_to_pi(st.yaw - ref_yaw)
        obs = {"idx": idx, "pos": np.array([st.x, st.y]), "yaw": st.yaw, "v": st.v,
               "ey": ey, "epsi": epsi, "prev_ey": prev_ey, "prev_epsi": prev_epsi}
        t0 = time.perf_counter()
        act = controller.control(obs, dt)
        compute_ms.append((time.perf_counter() - t0) * 1000.0)
        st = step_kinematic(st, act.steer, act.accel, dt, vp)
        idx2, s2, ey2, ref_yaw2 = track.project(np.array([st.x, st.y]))
        epsi2 = wrap_to_pi(st.yaw - ref_yaw2)
        if track.crossed_finish(prev_s, s2) and track.progress_delta(prev_s, s2) > 0 and k > 20:
            success = True
        prev_s = s2
        prev_ey = ey2
        prev_epsi = epsi2
        logger.log(
            time_s=k*dt, x_m=st.x, y_m=st.y, yaw_rad=st.yaw, speed_mps=st.v,
            steer_rad=act.steer, accel_mps2=act.accel,
            progress_m=s2, lateral_error_m=ey2, heading_error_rad=epsi2,
            compute_ms=compute_ms[-1],
        )
        if abs(ey2) > track.width/2 or success:
            break

    df = pd.DataFrame(logger.rows)
    step_path = EXP_DATA / f"{ctrl_name}_step_log.csv"
    logger.to_csv(step_path)
    save_series_and_trajectory(df, ctrl_name, EXP_FIG, track=track)

    lap_time = float(df["time_s"].iloc[-1]) if len(df) else timeout_s
    summary = {
        "controller": ctrl_name,
        "success": bool(success),
        "lap_time_s": lap_time if success else timeout_s if ctrl_name != "mpc" else lap_time,
        "mean_speed_mps": float(df["speed_mps"].mean()) if len(df) else 0.0,
        "rms_lateral_error_m": float(np.sqrt(np.mean(df["lateral_error_m"]**2))) if len(df) else 0.0,
        "mean_compute_ms": float(np.mean(compute_ms)) if compute_ms else 0.0,
        "progress_m": float(df["progress_m"].max()) if len(df) else 0.0,
        "step_csv": str(step_path),
    }
    write_summary(EXP_LOG / f"{ctrl_name}_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary

def train_rl(cfg: dict):
    dt = float(cfg["sim"]["dt"])
    env = Monitor(TrackRLEnv(dt=dt, target_speed=float(cfg["sim"].get("target_speed_mps", 12.0)),
                             timeout_s=float(cfg["sim"]["timeout_s"]),
                             seed=int(cfg["sim"].get("seed", 42))))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(EXP_LOG / "tb"),
        learning_rate=float(cfg["rl"].get("learning_rate", 3e-4)),
        n_steps=int(cfg["rl"].get("n_steps", 2048)),
        batch_size=int(cfg["rl"].get("batch_size", 64)),
        gamma=float(cfg["rl"].get("gamma", 0.99)),
        gae_lambda=float(cfg["rl"].get("gae_lambda", 0.95)),
        clip_range=float(cfg["rl"].get("clip_range", 0.2)),
    )
    metrics_csv = EXP_LOG / "rl_training_metrics.csv"
    callback = RLMetricCallback(dt=dt, save_csv_path=str(metrics_csv))
    model.learn(total_timesteps=int(cfg["rl"]["timesteps"]), callback=callback, progress_bar=False)
    model_path = MODELS / "ppo_track.zip"
    model.save(model_path)
    plot_csv(metrics_csv, EXP_FIG / "rl_training_summary.png")
    print(f"Saved model to {model_path}")
    return str(model_path)

def eval_rl(cfg: dict):
    model_path = MODELS / "ppo_track.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}. Run training first.")
    env = TrackRLEnv(dt=float(cfg["sim"]["dt"]),
                     target_speed=float(cfg["sim"].get("target_speed_mps", 12.0)),
                     timeout_s=float(cfg["sim"]["timeout_s"]),
                     seed=int(cfg["sim"].get("seed", 42)))
    model = PPO.load(model_path)
    obs, info = env.reset()
    logger = StepLogger()
    total_reward = 0.0
    success = False
    for k in range(env.max_steps):
        t0 = time.perf_counter()
        action, _ = model.predict(obs, deterministic=True)
        compute_ms = (time.perf_counter() - t0) * 1000.0
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        logger.log(
            time_s=k*env.dt, x_m=env.state.x, y_m=env.state.y, yaw_rad=env.state.yaw,
            speed_mps=env.state.v, steer_rad=float(action[0])*env.vp.max_steer,
            accel_mps2=float(action[1])*(env.vp.max_accel if action[1] >= 0 else env.vp.max_decel),
            progress_m=float(info.get("progress", 0.0)),
            lateral_error_m=float(info.get("ey", 0.0)),
            heading_error_rad=float(info.get("epsi", 0.0)),
            compute_ms=compute_ms,
            reward=float(reward),
        )
        if terminated or truncated:
            success = bool(info.get("success", False))
            break

    df = pd.DataFrame(logger.rows)
    step_path = EXP_DATA / "rl_step_log.csv"
    logger.to_csv(step_path)
    save_series_and_trajectory(df, "rl", EXP_FIG, track=env.track)
    lap_time = float(df["time_s"].iloc[-1]) if len(df) else float(cfg["sim"]["timeout_s"])
    summary = {
        "controller": "rl",
        "success": bool(success),
        "lap_time_s": lap_time if success else float(cfg["sim"]["timeout_s"]),
        "mean_speed_mps": float(df["speed_mps"].mean()) if len(df) else 0.0,
        "rms_lateral_error_m": float(np.sqrt(np.mean(df["lateral_error_m"]**2))) if len(df) else 0.0,
        "mean_compute_ms": float(df["compute_ms"].mean()) if len(df) else 0.0,
        "progress_m": float(df["progress_m"].max()) if len(df) else 0.0,
        "total_reward": float(total_reward),
        "step_csv": str(step_path),
    }
    write_summary(EXP_LOG / "rl_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="run", choices=["run","train","eval"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["controller"]["name"]
    if name == "rl":
        if args.mode == "train":
            train_rl(cfg)
        elif args.mode == "eval":
            eval_rl(cfg)
        else:
            # default run = eval if model exists else train+eval
            model_path = MODELS / "ppo_track.zip"
            if not model_path.exists():
                train_rl(cfg)
            eval_rl(cfg)
    else:
        run_tracking(cfg)

if __name__ == "__main__":
    main()
