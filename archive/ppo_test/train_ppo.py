import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from f1_track_env import F1TrackEnv

class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, metrics_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            episode_info = info.get('episode', {})
            row = {
                'episode': len(self.rows) + 1,
                'reward': float(episode_info.get('r', np.nan)),
                'length_steps': int(episode_info.get('l', 0)),
                'wall_time_s': float(episode_info.get('t', np.nan)),
                'lap_time_s': float(info.get('lap_time_s', np.nan)),
                'progress_m': float(info.get('progress_m', np.nan)),
                'track_length_m': float(info.get('track_length_m', np.nan)),
                'success': int(bool(info.get('success', False))),
                'laps': int(info.get('completed_laps', 0)),
                'speed_final_mps': float(info.get('speed_mps', np.nan)),
            }
            self.rows.append(row)
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        csv_path = self.metrics_dir / 'training_metrics.csv'
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

        episodes = np.array([r['episode'] for r in self.rows])
        rewards = np.array([r['reward'] for r in self.rows], dtype=float)
        success = np.array([r['success'] for r in self.rows], dtype=float)
        lap_times = np.array([r['lap_time_s'] for r in self.rows], dtype=float)
        progress = np.array([r['progress_m'] for r in self.rows], dtype=float)

        window = min(10, len(rewards))
        kernel = np.ones(window) / window
        reward_ma = np.convolve(rewards, kernel, mode='same')
        success_ma = np.convolve(success, kernel, mode='same')

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
        axes[0, 0].plot(episodes, rewards, alpha=0.45, label='episode reward')
        axes[0, 0].plot(episodes, reward_ma, linewidth=2.0, label=f'{window}-ep moving avg')
        axes[0, 0].set_title('Training reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()

        axes[0, 1].plot(episodes, success, alpha=0.5, label='success')
        axes[0, 1].plot(episodes, success_ma, linewidth=2.0, label='success rate')
        axes[0, 1].set_title('Success / completion')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success')
        axes[0, 1].set_ylim(-0.05, 1.05)
        axes[0, 1].legend()

        axes[1, 0].plot(episodes, lap_times, label='lap time')
        axes[1, 0].set_title('Episode time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Seconds')

        axes[1, 1].plot(episodes, progress, label='progress')
        axes[1, 1].set_title('Progress per episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Meters')

        fig.savefig(self.metrics_dir / 'training_summary.png', dpi=150)
        plt.close(fig)


def make_env(log_dir: str):
    def _init():
        env = F1TrackEnv(render_mode=None)
        return Monitor(env, filename=os.path.join(log_dir, 'monitor.csv'))
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--model-path', type=str, default='models/ppo_f1_track')
    parser.add_argument('--log-dir', type=str, default='runs/ppo_f1_track')
    args = parser.parse_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.model_path)).mkdir(parents=True, exist_ok=True)

    env_for_check = F1TrackEnv(render_mode=None)
    check_env(env_for_check, warn=True)
    env_for_check.close()

    env = DummyVecEnv([make_env(args.log_dir)])
    callback = EpisodeMetricsCallback(metrics_dir=args.log_dir)

    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=2e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None,
    )

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(args.model_path)
    env.close()
    print(f'saved model to {args.model_path}.zip')
    print(f'training metrics saved under: {args.log_dir}')


if __name__ == '__main__':
    main()