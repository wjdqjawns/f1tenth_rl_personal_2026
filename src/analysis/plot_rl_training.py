from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from .common import moving_average

def plot_csv(csv_path: str | Path, out_path: str | Path | None = None):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if out_path is None:
        out_path = Path("experiments/fig/rl_training_summary.png")
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ep = df["episode"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0,0].plot(ep, df["reward"], alpha=0.45, label="episode reward")
    axes[0,0].plot(ep, moving_average(df["reward"], 10), label="10-ep moving avg")
    axes[0,0].set_title("Training reward")
    axes[0,0].legend()

    axes[0,1].plot(ep, df["success"], alpha=0.5, label="success")
    axes[0,1].plot(ep, moving_average(df["success"], 10), label="success rate")
    axes[0,1].set_title("Success / completion")
    axes[0,1].legend()

    axes[0,2].plot(ep, df["lap_time_s"])
    axes[0,2].plot(ep, pd.Series(df["lap_time_s"]).cummin(), label="best lap so far")
    axes[0,2].set_title("Lap time")
    axes[0,2].legend()

    axes[1,0].plot(ep, df["episode_time_s"])
    axes[1,0].set_title("Episode time")

    axes[1,1].plot(ep, df["progress_m"])
    axes[1,1].set_title("Progress per episode")

    axes[1,2].plot(ep, df["mean_speed_mps"])
    axes[1,2].set_title("Mean speed per episode")

    for ax in axes.flat:
        ax.set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="experiments/log/rl_training_metrics.csv")
    parser.add_argument("--out", default="experiments/fig/rl_training_summary.png")
    args = parser.parse_args()
    plot_csv(args.csv, args.out)

if __name__ == "__main__":
    main()
