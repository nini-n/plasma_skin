# experiments/run_many_episodes.py

import numpy as np

from env.plasma_env import PlasmaEnv
from controllers.baseline_controller import BaselineController
from controllers.extremum_seeking import ExtremumSeekingController
from metrics.kpi import compute_episode_kpis, summarize_kpis


def run_episodes_for_controller(
    ControllerClass,
    controller_name,
    mode="A",
    n_episodes=20,
    base_seed=0,
    episode_length=200,
):
    """
    Run multiple episodes for a given controller class and scenario mode.
    Returns:
        kpi_list : list of KPI dicts (one per episode)
        summary  : dict with mean/std/min/max per KPI
    """
    kpi_list = []

    for ep in range(n_episodes):
        seed = base_seed + ep

        # --- Create environment for this episode ---
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

        # --- Create controller instance ---
        if ControllerClass is BaselineController:
            ctrl = BaselineController(
                V_min=env.V_min, V_max=env.V_max,
                f_min=env.f_min, f_max=env.f_max,
                D_min=env.D_min, D_max=env.D_max,
                seed=seed,
            )
        elif ControllerClass is ExtremumSeekingController:
            ctrl = ExtremumSeekingController(
                V_min=env.V_min, V_max=env.V_max,
                f_min=env.f_min, f_max=env.f_max,
                D_min=env.D_min, D_max=env.D_max,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported controller class: {ControllerClass}")

        # Reset env and controller
        obs = env.reset()
        u = ctrl.reset()

        reward = None
        info = None

        J_EM_list = []
        J_P_list = []
        violations_any = []

        for step in range(env.episode_length):
            # Controller uses previous reward/info (for ES); baseline ignores them.
            u = ctrl.compute_action(obs, reward=reward, info=info)

            obs, reward, done, info = env.step(u)

            metrics = info["metrics"]
            J_EM = metrics["J_EM"]
            J_P = metrics["J_P"]

            viol = info["violations"]
            any_v = int(
                viol["power"] or viol["current"] or viol["temp"] or viol["emi"]
            )

            J_EM_list.append(J_EM)
            J_P_list.append(J_P)
            violations_any.append(any_v)

            if done:
                break

        # KPIs for this episode
        kpis = compute_episode_kpis(
            J_EM=np.array(J_EM_list),
            J_P=np.array(J_P_list),
            violations_any=np.array(violations_any),
            P_rx_off=env.P_rx_off,
            P_p_max=env.P_p_max,
        )
        kpi_list.append(kpis)

    # Summarize over all episodes
    summary = summarize_kpis(kpi_list)

    print(
        f"\n=== {controller_name} (mode {mode}) over {n_episodes} episodes ==="
    )
    for key, stats in summary.items():
        mean = stats["mean"]
        std = stats["std"]
        min_v = stats["min"]
        max_v = stats["max"]
        print(
            f"{key:18s}: mean={mean:7.3f}, std={std:7.3f}, "
            f"min={min_v:7.3f}, max={max_v:7.3f}"
        )

    return kpi_list, summary


def main():
    mode = "C"        # <-- change to "B" or "C" for other scenarios
    n_episodes = 20
    episode_length = 200

    # Baseline controller
    run_episodes_for_controller(
        BaselineController,
        "BaselineController",
        mode=mode,
        n_episodes=n_episodes,
        base_seed=0,
        episode_length=episode_length,
    )

    # Extremum-Seeking controller
    run_episodes_for_controller(
        ExtremumSeekingController,
        "ExtremumSeekingController",
        mode=mode,
        n_episodes=n_episodes,
        base_seed=1000,
        episode_length=episode_length,
    )


if __name__ == "__main__":
    main()
