from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import COLOR_MAP, apply_plot_style, save_combined_runs_gif
from ..sim.track import Track

LOG_DIR = Path("experiments/log")
FIG_DIR = Path("experiments/fig")


def load_summaries(log_dir: Path):
    rows = []
    for p in sorted(log_dir.glob("*_summary.json")):
        if p.name.startswith("compare_runs"):
            continue
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            rows.append(obj)
    return pd.DataFrame(rows)


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{float(v):.3f}")
        else:
            out[c] = out[c].astype(str)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", action="store_true")
    args = parser.parse_args()

    apply_plot_style()
    df = load_summaries(LOG_DIR)
    if df.empty:
        raise SystemExit("No run summaries found in experiments/log")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_DIR / "compare_runs_summary.csv", index=False)

    ok = df[df["success"] == True].copy()
    if not ok.empty:
        agg = ok.groupby("controller").agg(
            lap_time_s_mean=("lap_time_s", "mean"),
            lap_time_s_std=("lap_time_s", "std"),
            mean_speed_mps_mean=("mean_speed_mps", "mean"),
            mean_speed_mps_std=("mean_speed_mps", "std"),
            rms_lateral_error_m_mean=("rms_lateral_error_m", "mean"),
            rms_lateral_error_m_std=("rms_lateral_error_m", "std"),
            mean_compute_ms_mean=("mean_compute_ms", "mean"),
            mean_compute_ms_std=("mean_compute_ms", "std"),
            runs=("controller", "size"),
        ).reset_index()
    else:
        agg = pd.DataFrame()
    agg.to_csv(LOG_DIR / "compare_runs_aggregate.csv", index=False)

    best_rows = []
    for ctrl, g in df.groupby("controller"):
        g_ok = g[g["success"] == True]
        row = g_ok.sort_values("lap_time_s").iloc[0] if not g_ok.empty else g.sort_values(["progress_m", "lap_time_s"], ascending=[False, True]).iloc[0]
        best_rows.append(row)
    best = pd.DataFrame(best_rows).reset_index(drop=True)
    best.to_csv(LOG_DIR / "compare_runs_best.csv", index=False)

    table_cols = ["controller", "lap_time_s", "mean_speed_mps", "rms_lateral_error_m", "mean_compute_ms", "success"]
    tdf = _format_table(best[table_cols])
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.axis("off")
    tbl = ax.table(cellText=tdf.values.tolist(), colLabels=list(tdf.columns), loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.5)
    ax.set_title("Best run summary per controller")
    fig.savefig(FIG_DIR / "compare_runs_summary_table.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, col, title in zip(
        axes.flat,
        ["lap_time_s", "mean_speed_mps", "rms_lateral_error_m", "mean_compute_ms"],
        ["Lap time [s]", "Mean speed [m/s]", "RMS lateral error [m]", "Mean compute [ms]"],
    ):
        ax.bar(best["controller"], best[col], color=[COLOR_MAP.get(c, 'tab:blue') for c in best['controller']])
        ax.set_title(title)
    fig.savefig(FIG_DIR / "compare_runs_bars.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    track = Track.procedural()
    ax.plot(track.left[:,0], track.left[:,1], '--', color='0.6', lw=1.1, label='Track left')
    ax.plot(track.right[:,0], track.right[:,1], '--', color='0.6', lw=1.1, label='Track right')
    ax.plot(track.center[:,0], track.center[:,1], ':', color='0.8', lw=1.0, label='Centerline')
    run_map = {}
    for _, row in best.iterrows():
        step_path = Path(row["step_csv"])
        if step_path.exists():
            sdf = pd.read_csv(step_path)
            ctrl = row["controller"]
            run_map[ctrl] = sdf
            ax.plot(sdf["x_m"], sdf["y_m"], label=ctrl, color=COLOR_MAP.get(ctrl, 'tab:blue'), lw=2)
    ax.set_title("Best trajectory per controller")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.savefig(FIG_DIR / "compare_runs_trajectory_overlay.png")
    plt.close(fig)

    if run_map:
        save_combined_runs_gif(run_map, FIG_DIR / "compare_runs_overlay.gif", track)

    print(f"Saved: {LOG_DIR/'compare_runs_summary.csv'}, {LOG_DIR/'compare_runs_aggregate.csv'}, {LOG_DIR/'compare_runs_best.csv'}")


if __name__ == "__main__":
    main()
