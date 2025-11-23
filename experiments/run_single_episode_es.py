# experiments/run_single_episode_es.py

import numpy as np
import matplotlib.pyplot as plt

from env.plasma_env import PlasmaEnv
from controllers.extremum_seeking import ExtremumSeekingController

def run_single_episode_es(seed=0):
    # --- Create environment ---
    env = PlasmaEnv(
        dt=0.05,
        episode_length=200,
        seed=seed,
        P_rx_off=1.0,
        P_p_max=10.0,
        I_max=5.0,
        T_ambient=25.0,
        T_max=70.0,
    )

    # --- Create extremum-seeking controller ---
    ctrl = ExtremumSeekingController(
        V_min=env.V_min, V_max=env.V_max,
        f_min=env.f_min, f_max=env.f_max,
        D_min=env.D_min, D_max=env.D_max,
        seed=seed,
        initial_step_scale=0.08,
        min_step_scale=0.01,
        step_decay=0.999,
    )

    # Reset env and controller
    obs = env.reset()
    u = ctrl.reset()

    # Storage for logging
    times = []
    J_EM_list = []
    J_P_list = []
    T_board_list = []
    reward_list = []

    t = 0.0
    reward = None
    info = None

    for step in range(env.episode_length):
        # Controller uses previous reward/info to update its internal state
        u = ctrl.compute_action(obs, reward=reward, info=info)

        # Environment step
        obs, reward, done, info = env.step(u)

        # Extract metrics
        metrics = info["metrics"]
        J_EM = metrics["J_EM"]
        J_P = metrics["J_P"]
        T_board = metrics["T_board"]

        # Log
        times.append(t)
        J_EM_list.append(J_EM)
        J_P_list.append(J_P)
        T_board_list.append(T_board)
        reward_list.append(reward)

        t += env.dt

        if done:
            break

    # Convert to numpy arrays
    times = np.array(times)
    J_EM_arr = np.array(J_EM_list)
    J_P_arr = np.array(J_P_list)
    T_board_arr = np.array(T_board_list)
    reward_arr = np.array(reward_list)

    # --- Simple KPIs ---
    avg_J_EM = np.mean(J_EM_arr)
    avg_J_P = np.mean(J_P_arr)
    avg_reward = np.mean(reward_arr)

    print("=== Single Episode Summary (Extremum-Seeking Controller) ===")
    print(f"Average J_EM:   {avg_J_EM:.3f}")
    print(f"Average J_P:    {avg_J_P:.3f}")
    print(f"Average reward: {avg_reward:.3f}")

    # --- Plots ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(times, J_EM_arr)
    axs[0].set_ylabel("J_EM (EM metric)")
    axs[0].grid(True)

    axs[1].plot(times, J_P_arr)
    axs[1].set_ylabel("J_P (plasma power)")
    axs[1].grid(True)

    axs[2].plot(times, T_board_arr)
    axs[2].set_ylabel("T_board (°C)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_single_episode_es(seed=0)
