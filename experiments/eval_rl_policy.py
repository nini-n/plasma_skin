# experiments/eval_rl_policy.py

import numpy as np
from env.plasma_env import PlasmaEnv
from controllers.rl_policy_gradient import PolicyGradientAgent
from metrics.kpi import compute_episode_kpis, summarize_kpis


def train_rl_agent(
    mode="A",
    n_episodes=100,
    episode_length=200,
    seed=0,
):
    """
    Train a PolicyGradientAgent on PlasmaEnv in the given mode.
    Returns the trained agent.
    """

    # Create an env to infer observation and action dimensions
    env = PlasmaEnv(
        dt=0.05,
        episode_length=episode_length,
        mode=mode,
        seed=seed,
    )

    obs0 = env.reset()
    obs_dim = obs0.shape[0]

    action_low = np.array([env.V_min, env.f_min, env.D_min], dtype=float)
    action_high = np.array([env.V_max, env.f_max, env.D_max], dtype=float)
    act_dim = action_low.shape[0]

    agent = PolicyGradientAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        lr=5e-3,
        gamma=0.99,
        init_log_std=-0.2,
        seed=seed,
    )

    for ep in range(n_episodes):
        # New env each episode (different random seed)
        env = PlasmaEnv(
            dt=0.05,
            episode_length=episode_length,
            mode=mode,
            seed=seed + ep,
        )

        obs = env.reset()
        agent.reset_episode()

        total_return = 0.0
        J_EM_list = []

        for t in range(episode_length):
            action = agent.select_action(obs)
            obs_next, reward, done, info = env.step(action)

            agent.store_step(obs, action, reward)

            total_return += reward
            J_EM_list.append(info["metrics"]["J_EM"])

            obs = obs_next

            if done:
                break

        agent.update_policy()

        print(
            f"[TRAIN mode {mode}] Episode {ep+1}/{n_episodes} | "
            f"Return: {total_return:.3f} | "
            f"Avg J_EM: {np.mean(J_EM_list):.3f}"
        )

    return agent


def evaluate_rl_agent(
    agent,
    mode="A",
    n_eval_episodes=20,
    episode_length=200,
    seed=1234,
):
    """
    Evaluate a trained PolicyGradientAgent over multiple episodes and
    compute KPIs using the same KPI module as for the other controllers.
    """

    kpi_list = []

    for ep in range(n_eval_episodes):
        env = PlasmaEnv(
            dt=0.05,
            episode_length=episode_length,
            mode=mode,
            seed=seed + ep,
        )

        obs = env.reset()

        J_EM_list = []
        J_P_list = []
        violations_any = []

        for t in range(episode_length):
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)

            metrics = info["metrics"]
            J_EM = metrics["J_EM"]
            J_P = metrics["J_P"]

            viol = info["violations"]
            any_v = int(viol["power"] or viol["current"] or viol["temp"] or viol["emi"])

            J_EM_list.append(J_EM)
            J_P_list.append(J_P)
            violations_any.append(any_v)

            if done:
                break

        # Compute KPIs for this episode
        kpis = compute_episode_kpis(
            J_EM=np.array(J_EM_list),
            J_P=np.array(J_P_list),
            violations_any=np.array(violations_any),
            P_rx_off=env.P_rx_off,
            P_p_max=env.P_p_max,
        )
        kpi_list.append(kpis)

    # Summarize KPIs across episodes
    summary = summarize_kpis(kpi_list)

    print(f"\n=== RL Policy (mode {mode}) over {n_eval_episodes} episodes ===")
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
    mode = "C"          # you can also try "B" or "C"
    n_train_episodes = 100
    n_eval_episodes = 20
    episode_length = 200
    seed = 0

    # 1) Train RL agent
    agent = train_rl_agent(
        mode=mode,
        n_episodes=n_train_episodes,
        episode_length=episode_length,
        seed=seed,
    )

    # 2) Evaluate trained policy with KPIs
    evaluate_rl_agent(
        agent=agent,
        mode=mode,
        n_eval_episodes=n_eval_episodes,
        episode_length=episode_length,
        seed=1000,
    )


if __name__ == "__main__":
    main()
