# Overview

This project compares Ackermann-style vehicle controllers on the same procedural closed track.

Tracking controllers:
- Pure Pursuit
- LQR
- sampling-based tracking MPC

Policy controller:
- PPO RL

Primary metrics:
- lap time
- success
- mean speed
- RMS lateral error
- mean compute time

RL additionally stores:
- reward curve
- success rate curve
- lap time curve
- progress curve
