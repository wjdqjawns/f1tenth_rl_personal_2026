# prj_f1_rl_fixed

Ackermann-vehicle track-control/RL comparison scaffold.

Included:
- Pure Pursuit baseline
- LQR tracking controller
- Lightweight sampling-based tracking MPC
- RL (PPO) training/evaluation
- RL training curves
- Run comparison with failed-run separation

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tensorboard
```

## Run controllers
```bash
python -m src.main --config config/pure_pursuit.yaml
python -m src.main --config config/lqr.yaml
python -m src.main --config config/mpc.yaml
```

## RL train / eval
```bash
python -m src.main --config config/rl.yaml --mode train
python -m src.main --config config/rl.yaml --mode eval
python -m src.analysis.plot_rl_training --csv experiments/log/rl_training_metrics.csv
```

## Compare runs
```bash
python -m src.analysis.compare_runs --existing
```

Notes:
- Pure Pursuit/LQR/MPC here are **tracking** controllers with a target speed.
- RL uses steer + accel directly and writes training curves to:
  - `experiments/log/rl_training_metrics.csv`
  - `experiments/fig/rl_training_summary.png`


## Added outputs
- `experiments/fig/<controller>_trajectory.gif`: each controller trajectory animation with cumulative path
- `compare_runs` now supports mixed string/numeric summary tables without rounding errors


## Plot styling and GIFs

This version uses SciencePlots when available.

- Install dependencies: `pip install -r requirements.txt`
- Generate comparison figures and overlay GIF: `python -m src.analysis.compare_runs --existing`

Created files:
- `experiments/fig/*_trajectory.gif`
- `experiments/fig/compare_runs_overlay.gif`
- `experiments/fig/compare_runs_summary_table.png`
