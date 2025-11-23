# experiments/run_rl_episodes.py

import numpy as np
import matplotlib.pyplot as plt

from env.plasma_env import PlasmaEnv
from controllers.rl_policy_gradient import PolicyGradientAgent


def run_rl_training(
    mode="A",
    n_episodes=50,
    episode_length=200,
    seed=0,
):
    """
    Train a simple policy gradient agent on PlasmaEnv
    for a given scenario mode ("A", "B", or "C").
    """

    # Create one environment to query obs/action dimensions and bounds
    env = PlasmaEnv(
        dt=0.05,
        episode_length=episode_length,
        mode=mode,
        seed=seed,
    )

    # Dummy reset to get initial observation shape
    obs0 = env.reset()
    obs_dim = obs0.shape[0]

    # Action bounds from env
    action_low = np.array([env.V_min, env.f_min, env.D_min], dtype=float)
    action_high = np.array([env.V_max, env.f_max, env.D_max], dtype=float)
    act_dim = action_low.shape[0]

    # Create the RL agent
    agent = PolicyGradientAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low=action_low,
        action_high=action_high,
        lr=1e-3,
        gamma=0.99,
        init_log_std=-0.5,
        seed=seed,
    )

    # Storage for episode returns and simple EM KPI
    episode_returns = []
    episode_avg_J_EM = []

    for ep in range(n_episodes):
        # Re-create env each episode to resample randomness
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
            # Select action from current policy
            action = agent.select_action(obs)

            # Step the environment
            obs_next, reward, done, info = env.step(action)

            # Store transition in agent buffers
            agent.store_step(obs, action, reward)

            # Logging
            total_return += reward
            J_EM_list.append(info["metrics"]["J_EM"])

            obs = obs_next

            if done:
                break

        # End of episode: update policy
        agent.update_policy()

        # Logging
        episode_returns.append(total_return)
        episode_avg_J_EM.append(np.mean(J_EM_list))

        print(
            f"[Mode {mode}] Episode {ep+1}/{n_episodes} | "
            f"Return: {total_return:.3f} | "
            f"Avg J_EM: {np.mean(J_EM_list):.3f}"
        )

    # Convert to arrays for plotting
    episode_returns = np.array(episode_returns)
    episode_avg_J_EM = np.array(episode_avg_J_EM)

    # Simple plots: how return and J_EM evolve over training
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(episode_returns)
    axs[0].set_ylabel("Episode return")
    axs[0].grid(True)

    axs[1].plot(episode_avg_J_EM)
    axs[1].set_ylabel("Avg J_EM per episode")
    axs[1].set_xlabel("Episode")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    return episode_returns, episode_avg_J_EM


if __name__ == "__main__":
    # Example: train in Scenario A, then you can try B and C
    run_rl_training(mode="A", n_episodes=50, episode_length=200, seed=0)
