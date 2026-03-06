import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from f1_track_env import F1TrackEnv


def _moving_mean(x, window=15):
    if len(x) < 2:
        return x
    w = min(window, len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')


def save_episode_plots(histories, metrics, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), tight_layout=True)

    for i, hist in enumerate(histories, start=1):
        t = hist['t']
        axes[0, 0].plot(t, hist['speed'], label=f'ep {i}', alpha=0.85)
        axes[0, 1].plot(t, np.rad2deg(hist['steer']), label=f'ep {i}', alpha=0.85)
        axes[1, 0].plot(t, hist['throttle_cmd'], label=f'ep {i}', alpha=0.85)
        axes[1, 1].plot(t, hist['reward'], label=f'ep {i}', alpha=0.75)
        axes[2, 0].plot(hist['x'], hist['y'], label=f'ep {i}', alpha=0.9)
        axes[2, 1].plot(t, hist['progress'], label=f'ep {i}', alpha=0.85)

    axes[0, 0].set_title('Speed vs time')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Speed [m/s]')

    axes[0, 1].set_title('Steering angle vs time')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Steer [deg]')

    axes[1, 0].set_title('Throttle / brake command vs time')
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Command [-1, 1]')

    axes[1, 1].set_title('Instant reward vs time')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Reward')

    axes[2, 0].set_title('Trajectories')
    axes[2, 0].set_xlabel('x [m]')
    axes[2, 0].set_ylabel('y [m]')
    axes[2, 0].axis('equal')

    axes[2, 1].set_title('Accumulated progress vs time')
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('Progress [m]')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    fig.savefig(out / 'episode_overlay.png', dpi=150)
    plt.close(fig)

    episodes = np.arange(1, len(metrics) + 1)
    rewards = np.array([m['reward'] for m in metrics], dtype=float)
    lap_times = np.array([m['lap_time_s'] for m in metrics], dtype=float)
    success = np.array([m['success'] for m in metrics], dtype=float)
    mean_speeds = np.array([m['mean_speed_mps'] for m in metrics], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)
    axes[0, 0].plot(episodes, rewards, marker='o', label='reward')
    axes[0, 0].plot(episodes, _moving_mean(rewards), linewidth=2, label='smoothed')
    axes[0, 0].set_title('Episode reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, lap_times, marker='o')
    axes[0, 1].set_title('Lap / episode time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Time [s]')

    axes[1, 0].plot(episodes, success, marker='o')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title('Success rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success')

    axes[1, 1].plot(episodes, mean_speeds, marker='o')
    axes[1, 1].set_title('Mean speed per episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Mean speed [m/s]')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.savefig(out / 'episode_metrics.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/ppo_f1_track.zip')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='eval_outputs')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    env = F1TrackEnv(render_mode='human' if args.render else None)
    model = PPO.load(args.model_path)

    histories = []
    metrics = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if args.render:
                env.render()
                time.sleep(env.dt * 0.35)
            done = terminated or truncated

        hist = env.get_episode_history()
        histories.append(hist)
        mean_speed = float(np.mean(hist['speed'])) if len(hist['speed']) else 0.0
        metric = {
            'episode': ep + 1,
            'reward': float(total_reward),
            'lap_time_s': float(info.get('lap_time_s', len(hist['t']) * env.dt)),
            'success': bool(info.get('success', False)),
            'laps': int(info.get('completed_laps', 0)),
            'progress_m': float(info.get('progress_m', 0.0)),
            'track_length_m': float(info.get('track_length_m', 1.0)),
            'mean_speed_mps': mean_speed,
            'final_speed_mps': float(info.get('speed_mps', 0.0)),
        }
        metrics.append(metric)
        print(
            f"episode={metric['episode']}, reward={metric['reward']:.2f}, "
            f"success={metric['success']}, laps={metric['laps']}, "
            f"lap_time={metric['lap_time_s']:.2f}s, mean_speed={metric['mean_speed_mps']:.2f}m/s, "
            f"progress={metric['progress_m']:.1f}/{metric['track_length_m']:.1f}m"
        )

    save_episode_plots(histories, metrics, args.out_dir)
    env.close()
    print(f'plots saved to: {os.path.abspath(args.out_dir)}')

if __name__ == '__main__':
    main()