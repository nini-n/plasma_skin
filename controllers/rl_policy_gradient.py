# controllers/rl_policy_gradient.py

import numpy as np


class PolicyGradientAgent:
    """
    Improved REINFORCE policy gradient agent with a Gaussian policy
    defined in a normalized action space a_norm in [-1, 1]^act_dim.

        a_norm ~ N(mean, diag(std^2))
        mean   = tanh(W @ obs + b)

    The normalized action a_norm is then mapped to the physical
    action space using:

        action = 0.5 * (high + low) + 0.5 * (high - low) * a_norm

    This avoids issues with very different physical scales across
    action dimensions (e.g. volts vs Hz).
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        action_low,
        action_high,
        lr=5e-3,          # slightly larger learning rate
        gamma=0.99,
        init_log_std=-0.2,  # std ~ 0.82 in normalized space
        seed=None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Physical action bounds (env units)
        self.action_low = np.asarray(action_low, dtype=float)
        self.action_high = np.asarray(action_high, dtype=float)

        # Precompute mid and span for mapping [-1,1] -> [low, high]
        self.action_mid = 0.5 * (self.action_high + self.action_low)
        self.action_span = 0.5 * (self.action_high - self.action_low)

        self.lr = lr
        self.gamma = gamma

        self.rng = np.random.default_rng(seed)

        # Policy parameters for mean in normalized space
        # mean = tanh(W @ obs + b), so W: (act_dim, obs_dim)
        self.W = 0.01 * self.rng.normal(size=(act_dim, obs_dim))
        self.b = np.zeros(act_dim, dtype=float)

        # Log std for each action dimension (normalized space)
        self.log_std = np.full(act_dim, init_log_std, dtype=float)

        # Episode buffers (store normalized actions)
        self.obs_buf = []
        self.act_norm_buf = []
        self.rew_buf = []

    # ------------------------------------------------------------------
    # Episode buffer management
    # ------------------------------------------------------------------

    def reset_episode(self):
        """Clear episode buffers before starting a new rollout."""
        self.obs_buf = []
        self.act_norm_buf = []
        self.rew_buf = []

    def store_step(self, obs, action, reward):
        """
        Store one transition (obs, action, reward) in the buffers.

        Here, 'action' is expected to be the PHYSICAL action applied
        to the environment, so we first convert it back to the
        normalized space, a_norm in [-1, 1]^act_dim.
        """
        obs = np.asarray(obs, dtype=float)
        action = np.asarray(action, dtype=float)

        # Map physical action back to normalized space
        # a_norm = (action - mid) / span
        a_norm = (action - self.action_mid) / (self.action_span + 1e-8)
        # Clip for numerical safety
        a_norm = np.clip(a_norm, -1.0, 1.0)

        self.obs_buf.append(obs.copy())
        self.act_norm_buf.append(a_norm.copy())
        self.rew_buf.append(float(reward))

    # ------------------------------------------------------------------
    # Policy and action sampling
    # ------------------------------------------------------------------

    def _get_mean_std(self, obs):
        """
        Compute mean and std of the Gaussian policy in normalized space.
        mean = tanh(W @ obs + b) in [-1, 1]^act_dim
        """
        obs = np.asarray(obs, dtype=float)
        z = self.W @ obs + self.b           # pre-activation
        mean = np.tanh(z)                   # in [-1, 1]
        std = np.exp(self.log_std)          # positive
        return mean, std, z

    def select_action(self, obs):
        """
        Sample a normalized action from the current policy and map it
        to the physical action space.

        Returns:
            action_phys: np.ndarray in [action_low, action_high]
        """
        mean, std, z = self._get_mean_std(obs)
        noise = self.rng.normal(size=self.act_dim)
        a_norm = mean + std * noise

        # Clip normalized action to [-1, 1]
        a_norm = np.clip(a_norm, -1.0, 1.0)

        # Map to physical space
        action_phys = self.action_mid + self.action_span * a_norm

        return action_phys

    # ------------------------------------------------------------------
    # Returns and policy update (REINFORCE)
    # ------------------------------------------------------------------

    def _compute_returns(self):
        """
        Compute discounted returns G_t for an episode using gamma.
        Returns are normalized to reduce variance.
        """
        T = len(self.rew_buf)
        returns = np.zeros(T, dtype=float)
        G = 0.0
        for t in reversed(range(T)):
            G = self.rew_buf[t] + self.gamma * G
            returns[t] = G

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update_policy(self):
        """
        REINFORCE update:
            grad_theta J ≈ E[ G_t * grad_theta log pi(a_t | obs_t) ]
        with a Gaussian policy defined in normalized action space.

        We perform one gradient ascent step on W, b, and log_std.
        """
        if len(self.rew_buf) == 0:
            return

        obs_arr = np.asarray(self.obs_buf, dtype=float)           # (T, obs_dim)
        act_norm_arr = np.asarray(self.act_norm_buf, dtype=float) # (T, act_dim)
        returns = self._compute_returns()                         # (T,)

        std = np.exp(self.log_std)                                # (act_dim,)
        var = std ** 2

        # Gradients
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_log_std = np.zeros_like(self.log_std)

        T = len(returns)
        for t in range(T):
            obs_t = obs_arr[t]
            a_norm_t = act_norm_arr[t]
            G_t = returns[t]

            # Forward for this obs
            z_t = self.W @ obs_t + self.b
            mean_t = np.tanh(z_t)

            diff = a_norm_t - mean_t  # (act_dim,)

            # grad log pi wrt mean in normalized space
            grad_logp_mean = diff / (var + 1e-8)

            # chain rule: mean = tanh(z), so dmean/dz = 1 - tanh(z)^2
            dmean_dz = 1.0 - mean_t ** 2
            grad_z = grad_logp_mean * dmean_dz  # (act_dim,)

            # Accumulate gradients for W and b
            grad_W += np.outer(grad_z, obs_t) * G_t
            grad_b += grad_z * G_t

            # grad log pi wrt log_std:
            # log pi ~ -0.5 * sum( (diff^2 / var) + 2*log_std + const )
            # d log pi / d log_std_j = (diff_j^2 / var_j) - 1
            grad_logp_log_std = (diff ** 2) / (var + 1e-8) - 1.0
            grad_log_std += grad_logp_log_std * G_t

        # Gradient ascent step
        scale = self.lr / float(T)
        self.W += scale * grad_W
        self.b += scale * grad_b
        self.log_std += scale * grad_log_std

        # Clear buffers after update
        self.reset_episode()
