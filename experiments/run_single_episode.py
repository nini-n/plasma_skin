# experiments/run_single_episode.py

import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust these imports according to your project structure
from env.plasma_env import PlasmaEnv
from controllers.baseline_controller import BaselineController

def run_single_episode(seed=0):
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

    # --- Create baseline controller ---
    ctrl = BaselineController(
        V_min=env.V_min, V_max=env.V_max,
        f_min=env.f_min, f_max=env.f_max,
        D_min=env.D_min, D_max=env.D_max,
        seed=seed
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
    power_viol = []
    current_viol = []
    temp_viol = []

    t = 0.0

    for step in range(env.episode_length):
        # Controller computes action based on obs
        u = ctrl.compute_action(obs)

        # Environment step
        obs, reward, done, info = env.step(u)

        # Extract metrics
        metrics = info["metrics"]
        J_EM = metrics["J_EM"]
        J_P = metrics["J_P"]
        T_board = metrics["T_board"]

        viol = info["violations"]
        pv = int(viol["power"])
        cv = int(viol["current"])
        tv = int(viol["temp"])

        # Log
        times.append(t)
        J_EM_list.append(J_EM)
        J_P_list.append(J_P)
        T_board_list.append(T_board)
        reward_list.append(reward)
        power_viol.append(pv)
        current_viol.append(cv)
        temp_viol.append(tv)

        t += env.dt

        if done:
            break

    # Convert to numpy arrays
    times = np.array(times)
    J_EM_arr = np.array(J_EM_list)
    J_P_arr = np.array(J_P_list)
    T_board_arr = np.array(T_board_list)
    reward_arr = np.array(reward_list)
    power_viol = np.array(power_viol)
    current_viol = np.array(current_viol)
    temp_viol = np.array(temp_viol)

    # --- Simple KPIs ---
    avg_J_EM = np.mean(J_EM_arr)
    avg_J_P = np.mean(J_P_arr)
    avg_reward = np.mean(reward_arr)
    R_safe = 1.0 - np.mean(np.maximum(power_viol, np.maximum(current_viol, temp_viol)))

    print("=== Single Episode Summary (Baseline Controller) ===")
    print(f"Average J_EM:   {avg_J_EM:.3f}")
    print(f"Average J_P:    {avg_J_P:.3f}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Safe operation ratio R_safe: {R_safe:.3f}")

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
    run_single_episode(seed=0)
