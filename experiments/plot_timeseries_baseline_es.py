# experiments/plot_timeseries_baseline_es.py

import numpy as np
import matplotlib.pyplot as plt

from env.plasma_env import PlasmaEnv
from controllers.baseline_controller import BaselineController
from controllers.extremum_seeking import ExtremumSeekingController


def simulate_episode(controller_cls, mode="A", seed=0, episode_length=200):
    """
    Run a single episode for the given controller class and return time-series logs.
    """
    env = PlasmaEnv(
        dt=0.05,
        episode_length=episode_length,
        mode=mode,
        seed=seed,
        P_rx_off=1.0,
        P_p_max=10.0,
        I_max=5.0,
        T_ambient=25.0,
        T_max=70.0,
    )

    # Instantiate controller
    if controller_cls is BaselineController:
        ctrl = BaselineController(
            V_min=env.V_min, V_max=env.V_max,
            f_min=env.f_min, f_max=env.f_max,
            D_min=env.D_min, D_max=env.D_max,
            seed=seed,
        )
        name = "Baseline"
    elif controller_cls is ExtremumSeekingController:
        ctrl = ExtremumSeekingController(
            V_min=env.V_min, V_max=env.V_max,
            f_min=env.f_min, f_max=env.f_max,
            D_min=env.D_min, D_max=env.D_max,
            seed=seed,
        )
        name = "ES"
    else:
        raise ValueError("Unsupported controller class")

    obs = env.reset()
    u = ctrl.reset()

    t = 0.0
    reward = None
    info = None

    times = []
    J_EM_list = []
    J_P_list = []
    T_board_list = []

    for step in range(env.episode_length):
        u = ctrl.compute_action(obs, reward=reward, info=info)
        obs, reward, done, info = env.step(u)

        metrics = info["metrics"]
        J_EM_list.append(metrics["J_EM"])
        J_P_list.append(metrics["J_P"])
        T_board_list.append(metrics["T_board"])
        times.append(t)

        t += env.dt
        if done:
            break

    logs = {
        "name": name,
        "t": np.array(times),
        "J_EM": np.array(J_EM_list),
        "J_P": np.array(J_P_list),
        "T_board": np.array(T_board_list),
    }
    return logs


def plot_timeseries(mode="A", seed_baseline=0, seed_es=0, episode_length=200):
    """
    Generate a time-series figure comparing Baseline and ES controllers
    in the given scenario mode.
    """
    logs_baseline = simulate_episode(
        BaselineController, mode=mode, seed=seed_baseline, episode_length=episode_length
    )
    logs_es = simulate_episode(
        ExtremumSeekingController, mode=mode, seed=seed_es, episode_length=episode_length
    )

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # J_EM
    axs[0].plot(logs_baseline["t"], logs_baseline["J_EM"], label="Baseline")
    axs[0].plot(logs_es["t"], logs_es["J_EM"], label="ES")
    axs[0].set_ylabel("J_EM (EM metric)")
    axs[0].legend()
    axs[0].grid(True)

    # J_P
    axs[1].plot(logs_baseline["t"], logs_baseline["J_P"], label="Baseline")
    axs[1].plot(logs_es["t"], logs_es["J_P"], label="ES")
    axs[1].set_ylabel("J_P (plasma power)")
    axs[1].grid(True)

    # T_board
    axs[2].plot(logs_baseline["t"], logs_baseline["T_board"], label="Baseline")
    axs[2].plot(logs_es["t"], logs_es["T_board"], label="ES")
    axs[2].set_ylabel("T_board (°C)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)

    fig.suptitle(f"Time-series comparison (Scenario {mode})")

    # Add a bit more left margin so the y-axis label is fully visible
    plt.tight_layout(rect=[0.15, 0.03, 1, 0.95])

    # Save and show
    out_path = f"time_series_baseline_vs_es_mode_{mode}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")
    plt.show()

if __name__ == "__main__":
    # Scenario A by default
    plot_timeseries(mode="A", seed_baseline=0, seed_es=0, episode_length=200)
