# experiments/plot_rl_training_curves.py

import numpy as np
import matplotlib.pyplot as plt

from env.plasma_env import PlasmaEnv
from controllers.rl_policy_gradient import PolicyGradientAgent


def run_rl_training_with_logging(
    mode="A",
    n_episodes=100,
    episode_length=200,
    seed=0,
):
    """
    Train a PolicyGradientAgent on PlasmaEnv and log per-episode
    returns and average J_EM for plotting.
    """
    # Env to infer dimensions
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

    episode_returns = []
    episode_avg_J_EM = []

    for ep in range(n_episodes):
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

        episode_returns.append(total_return)
        episode_avg_J_EM.append(np.mean(J_EM_list))

        print(
            f"[RL TRAIN mode {mode}] Episode {ep+1}/{n_episodes} | "
            f"Return: {total_return:.3f} | "
            f"Avg J_EM: {np.mean(J_EM_list):.3f}"
        )

    return np.array(episode_returns), np.array(episode_avg_J_EM)


def plot_rl_training_curves(mode="A", n_episodes=100, episode_length=200, seed=0):
    """
    Run RL training and generate a two-panel figure showing
    episode return and average J_EM vs. episode index.
    """
    returns, avg_J_EM = run_rl_training_with_logging(
        mode=mode,
        n_episodes=n_episodes,
        episode_length=episode_length,
        seed=seed,
    )

    episodes = np.arange(1, len(returns) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(episodes, returns)
    axs[0].set_ylabel("Episode return")
    axs[0].grid(True)

    axs[1].plot(episodes, avg_J_EM)
    axs[1].set_ylabel("Avg J_EM per episode")
    axs[1].set_xlabel("Episode")
    axs[1].grid(True)

    fig.suptitle(f"RL training curves (Scenario {mode})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"rl_training_curves_mode_{mode}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")
    plt.show()


if __name__ == "__main__":
    # Scenario A by default
    plot_rl_training_curves(mode="A", n_episodes=100, episode_length=200, seed=0)
