from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

COLOR_MAP = {
    "pure_pursuit": "tab:blue",
    "lqr": "tab:green",
    "mpc": "tab:orange",
    "rl": "tab:red",
}
GIF_RGB = {
    "pure_pursuit": (31, 119, 180),
    "lqr": (44, 160, 44),
    "mpc": (255, 127, 14),
    "rl": (214, 39, 40),
}


def apply_plot_style() -> None:
    """Use SciencePlots when available; fall back gracefully."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "grid", "no-latex"])
    except Exception:
        plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


apply_plot_style()


def moving_average(x, w=10):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        j0 = max(0, i-w+1)
        out[i] = np.mean(x[j0:i+1])
    return out


def _fig_color(prefix: str) -> str:
    return COLOR_MAP.get(prefix, "tab:blue")


def _all_points(df: pd.DataFrame, track=None):
    pts = [df[["x_m", "y_m"]].to_numpy(dtype=float)]
    if track is not None:
        pts += [np.asarray(track.left, dtype=float), np.asarray(track.right, dtype=float), np.asarray(track.center, dtype=float)]
    arr = np.vstack(pts)
    return arr[:, 0], arr[:, 1]


def save_series_and_trajectory(df: pd.DataFrame, prefix: str, fig_dir: str | Path, track=None):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    color = _fig_color(prefix)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0,0].plot(df["time_s"], df["speed_mps"], color=color, lw=2)
    axes[0,0].set_title("Speed")
    axes[0,0].set_ylabel("m/s")
    axes[0,1].plot(df["time_s"], df["steer_rad"], color=color, lw=2)
    axes[0,1].set_title("Steering")
    axes[0,1].set_ylabel("rad")
    axes[1,0].plot(df["time_s"], df["progress_m"], color=color, lw=2)
    axes[1,0].set_title("Progress")
    axes[1,0].set_ylabel("m")
    axes[1,1].plot(df["time_s"], df["lateral_error_m"], color=color, lw=2)
    axes[1,1].set_title("Lateral error")
    axes[1,1].set_ylabel("m")
    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
    fig.savefig(fig_dir / f"{prefix}_metrics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,7), constrained_layout=True)
    if track is not None:
        ax.plot(track.left[:,0], track.left[:,1], '--', color='0.55', lw=1.2, label='Track left')
        ax.plot(track.right[:,0], track.right[:,1], '--', color='0.55', lw=1.2, label='Track right')
        ax.plot(track.center[:,0], track.center[:,1], ':', color='0.75', lw=1.2, label='Centerline')
    ax.plot(df["x_m"], df["y_m"], color=color, lw=2.4, label="Trajectory")
    ax.scatter(df["x_m"].iloc[0], df["y_m"].iloc[0], color='black', s=28, label='Start', zorder=3)
    ax.set_title(f"{prefix} trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc='best')
    fig.savefig(fig_dir / f"{prefix}_trajectory.png")
    plt.close(fig)

    if track is not None and len(df) > 1:
        save_run_gif(df, prefix, fig_dir / f"{prefix}_trajectory.gif", track)


def _to_px(arr: np.ndarray, bounds, size: int = 720, pad: int = 32):
    x_min, x_max, y_min, y_max = bounds
    sx = (size - 2*pad) / max(x_max - x_min, 1e-9)
    sy = (size - 2*pad) / max(y_max - y_min, 1e-9)
    s = min(sx, sy)
    x = pad + (arr[:, 0] - x_min) * s
    y = size - (pad + (arr[:, 1] - y_min) * s)
    return np.column_stack([x, y]).astype(int)


def _draw_car(draw: ImageDraw.ImageDraw, x: int, y: int, dx: float, dy: float, color):
    draw.ellipse((x-6, y-6, x+6, y+6), fill=color, outline=(0,0,0))
    n = max((dx*dx + dy*dy) ** 0.5, 1.0)
    hx = int(x + 14 * dx / n)
    hy = int(y + 14 * dy / n)
    draw.line((x, y, hx, hy), fill=(0, 0, 0), width=2)


def save_run_gif(df: pd.DataFrame, prefix: str, gif_path: str | Path, track, frame_stride: int = 4):
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    size = 720
    bg = (255, 255, 255)
    line_color = GIF_RGB.get(prefix, (31, 119, 180))
    trail_color = tuple(min(255, c + 35) for c in line_color)

    xs, ys = _all_points(df, track)
    margin = 2.0
    bounds = (float(xs.min()-margin), float(xs.max()+margin), float(ys.min()-margin), float(ys.max()+margin))
    left_px = _to_px(np.asarray(track.left, dtype=float), bounds, size=size)
    right_px = _to_px(np.asarray(track.right, dtype=float), bounds, size=size)
    center_px = _to_px(np.asarray(track.center, dtype=float), bounds, size=size)
    traj_px = _to_px(df[["x_m", "y_m"]].to_numpy(dtype=float), bounds, size=size)

    frames = []
    idxs = list(range(1, len(traj_px), max(1, frame_stride)))
    if idxs and idxs[-1] != len(traj_px)-1:
        idxs.append(len(traj_px)-1)
    elif not idxs:
        idxs = [len(traj_px)-1]

    for i in idxs:
        im = Image.new('RGB', (size, size), bg)
        dr = ImageDraw.Draw(im)
        dr.line([tuple(p) for p in left_px], fill=(120, 120, 120), width=3)
        dr.line([tuple(p) for p in right_px], fill=(120, 120, 120), width=3)
        dr.line([tuple(p) for p in center_px], fill=(220, 220, 220), width=1)
        if i > 1:
            dr.line([tuple(p) for p in traj_px[:i+1]], fill=trail_color, width=4)
        x, y = traj_px[i]
        j0 = max(0, i-1)
        dx = traj_px[i][0] - traj_px[j0][0]
        dy = traj_px[i][1] - traj_px[j0][1]
        _draw_car(dr, x, y, dx, dy, line_color)
        dr.text((18, 16), f"{prefix}", fill=(0,0,0))
        dr.text((18, 40), f"t={df['time_s'].iloc[i]:.1f}s", fill=(0,0,0))
        frames.append(im)

    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=80, loop=0)


def save_combined_runs_gif(run_map: dict[str, pd.DataFrame], gif_path: str | Path, track, frame_stride: int = 4):
    """Create one cumulative GIF with multiple controllers overlaid."""
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    size = 760
    bg = (255, 255, 255)

    pts = []
    for df in run_map.values():
        pts.append(df[["x_m", "y_m"]].to_numpy(dtype=float))
    pts += [np.asarray(track.left, dtype=float), np.asarray(track.right, dtype=float), np.asarray(track.center, dtype=float)]
    arr = np.vstack(pts)
    margin = 2.0
    bounds = (float(arr[:,0].min()-margin), float(arr[:,0].max()+margin), float(arr[:,1].min()-margin), float(arr[:,1].max()+margin))

    left_px = _to_px(np.asarray(track.left, dtype=float), bounds, size=size)
    right_px = _to_px(np.asarray(track.right, dtype=float), bounds, size=size)
    center_px = _to_px(np.asarray(track.center, dtype=float), bounds, size=size)
    traj_px = {k: _to_px(v[["x_m", "y_m"]].to_numpy(dtype=float), bounds, size=size) for k, v in run_map.items()}
    max_len = max(len(v) for v in traj_px.values())
    idxs = list(range(1, max_len, max(1, frame_stride)))
    if not idxs or idxs[-1] != max_len-1:
        idxs.append(max_len-1)

    frames = []
    for i in idxs:
        im = Image.new('RGB', (size, size), bg)
        dr = ImageDraw.Draw(im)
        dr.line([tuple(p) for p in left_px], fill=(120, 120, 120), width=3)
        dr.line([tuple(p) for p in right_px], fill=(120, 120, 120), width=3)
        dr.line([tuple(p) for p in center_px], fill=(220, 220, 220), width=1)

        y_text = 16
        for name, px in traj_px.items():
            color = GIF_RGB.get(name, (31, 119, 180))
            trail_color = tuple(min(255, c + 35) for c in color)
            j = min(i, len(px)-1)
            if j > 1:
                dr.line([tuple(p) for p in px[:j+1]], fill=trail_color, width=4)
            x, y = px[j]
            j0 = max(0, j-1)
            dx = px[j][0] - px[j0][0]
            dy = px[j][1] - px[j0][1]
            _draw_car(dr, x, y, dx, dy, color)
            dr.rectangle((16, y_text, 30, y_text+12), fill=color, outline=(0,0,0))
            dr.text((36, y_text-2), name, fill=(0,0,0))
            y_text += 20

        # show common comparison time based on max available step time
        ts = []
        for name, df in run_map.items():
            j = min(i, len(df)-1)
            ts.append(float(df['time_s'].iloc[j]))
        dr.text((16, size-32), f"comparison time ~ {max(ts):.1f}s", fill=(0,0,0))
        frames.append(im)

    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=90, loop=0)
