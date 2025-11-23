# controllers/baseline_controller.py

import numpy as np

class BaselineController:
    """
    Very simple baseline controller:
    - Starts from a safe mid-range action.
    - Adds small random perturbations at each step (within bounds).
    """

    def __init__(
        self,
        V_min, V_max,
        f_min, f_max,
        D_min, D_max,
        seed=None,
        noise_scale_V=0.1,
        noise_scale_f=0.1,
        noise_scale_D=0.05,
    ):
        self.V_min, self.V_max = V_min, V_max
        self.f_min, self.f_max = f_min, f_max
        self.D_min, self.D_max = D_min, D_max

        self.noise_scale_V = noise_scale_V
        self.noise_scale_f = noise_scale_f
        self.noise_scale_D = noise_scale_D

        self.rng = np.random.default_rng(seed)

        # initial nominal action (mid-range)
        V0 = 0.5 * (V_min + V_max)
        f0 = 0.5 * (f_min + f_max)
        D0 = 0.3  # modest duty
        self.u_nominal = np.array([V0, f0, D0], dtype=float)

    def reset(self):
        """
        Reset any internal state of the controller.
        Here we just reset to the nominal action.
        """
        return self.u_nominal.copy()

    def compute_action(self, obs, reward=None, info=None):
        """
        Compute next action given observation.
        Baseline: nominal action + small random noise, then clipped.
        reward/info are ignored (for API compatibility).
        """
        # Small zero-mean noise on each component
        dV = self.noise_scale_V * (self.V_max - self.V_min) * self.rng.normal()
        df = self.noise_scale_f * (self.f_max - self.f_min) * self.rng.normal()
        dD = self.noise_scale_D * (self.D_max - self.D_min) * self.rng.normal()

        u = self.u_nominal + np.array([dV, df, dD])

        # Clip to bounds
        u[0] = np.clip(u[0], self.V_min, self.V_max)
        u[1] = np.clip(u[1], self.f_min, self.f_max)
        u[2] = np.clip(u[2], self.D_min, self.D_max)

        return u

