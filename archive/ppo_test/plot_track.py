import matplotlib.pyplot as plt
import numpy as np

from f1_track_env import F1TrackEnv


def main():
    env = F1TrackEnv(render_mode=None)
    plt.figure(figsize=(9, 7))
    plt.plot(env.left_boundary[:, 0], env.left_boundary[:, 1], label='left boundary')
    plt.plot(env.right_boundary[:, 0], env.right_boundary[:, 1], label='right boundary')
    plt.plot(env.track_center[:, 0], env.track_center[:, 1], '--', label='reference centerline')
    plt.plot([env.start_line_a[0], env.start_line_b[0]], [env.start_line_a[1], env.start_line_b[1]], label='start/finish')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Procedural closed racing track')
    plt.show()

if __name__ == '__main__':
    main()