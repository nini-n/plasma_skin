import numpy as np


class PlasmaEnv:
    """
    Simple virtual environment for an adaptive plasma skin.

    - Action: u = [V_HV, f_HV, D]
    - Observations: normalized EM metric, power metric, temperature,
                    normalized action, constraint flags
    - API: reset(), step(action)

    Scenarios (mode):
        "A": baseline, low noise, slow drift
        "B": high-noise measurements
        "C": stronger EM drift
    """

    def __init__(
        self,
        dt=0.05,                 # simulation time step [s]
        episode_length=200,      # number of steps per episode
        V_min=0.5, V_max=5.0,    # kV
        f_min=5e3, f_max=30e3,   # Hz
        D_min=0.0, D_max=1.0,    # 0..1
        P_rx_off=1.0,            # reference EM power (arbitrary units)
        P_p_max=10.0,            # max allowed plasma power (arb. units)
        I_max=5.0,               # max HV current (arb. units)
        T_ambient=25.0,          # ambient temperature [°C]
        T_max=70.0,              # max allowed board temp [°C]
        noise_level_em=0.02,
        noise_level_p=0.02,
        mode="A",                # scenario selector: "A", "B", "C"
        seed=None,
    ):
        self.dt = dt
        self.episode_length = episode_length

        # Action bounds
        self.V_min, self.V_max = V_min, V_max
        self.f_min, self.f_max = f_min, f_max
        self.D_min, self.D_max = D_min, D_max

        # Reference & limits
        self.P_rx_off = P_rx_off
        self.P_p_max = P_p_max
        self.I_max = I_max
        self.T_ambient = T_ambient
        self.T_max = T_max

        # Scenario selection
        self.mode = mode
        if mode == "A":
            # Baseline: low noise, slow drift
            self.noise_level_em = noise_level_em
            self.noise_level_p = noise_level_p
            self.drift_rate = 0.001
        elif mode == "B":
            # High-noise scenario (more measurement noise on EM and power)
            self.noise_level_em = noise_level_em * 3.0
            self.noise_level_p = noise_level_p * 3.0
            self.drift_rate = 0.001
        elif mode == "C":
            # Stronger drift scenario (plant slowly changes more)
            self.noise_level_em = noise_level_em
            self.noise_level_p = noise_level_p
            self.drift_rate = 0.005
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'A', 'B' or 'C'.")

        # RNG
        self.rng = np.random.default_rng(seed)

        # Internal state placeholders
        # state = [em_state, p_state, T_board, drift]
        self.step_count = 0
        self.state = None
        self.prev_action = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset environment to an initial state.
        Returns:
            obs_0: initial observation (np.ndarray)
        """
        self.step_count = 0

        # em_state, p_state, board_temp, drift
        em_state = 1.0
        p_state = 0.0
        T_board = self.T_ambient
        drift = 0.0

        self.state = np.array([em_state, p_state, T_board, drift], dtype=float)

        # initial safe action (mid-range)
        V0 = 0.5 * (self.V_min + self.V_max)
        f0 = 0.5 * (self.f_min + self.f_max)
        D0 = 0.3  # modest duty
        self.prev_action = np.array([V0, f0, D0], dtype=float)

        obs = self._build_observation(
            action=self.prev_action,
            state=self.state,
            violation_flags=[0, 0, 0, 0],
        )
        return obs

    def step(self, action):
        """
        One simulation step.

        Input:
            action: np.array([V_HV, f_HV, D])

        Returns:
            obs    : next observation (np.ndarray)
            reward : scalar reward (float), typically -J_aug
            done   : bool, episode termination flag
            info   : dict with raw metrics and violation flags
        """
        self.step_count += 1

        # Clamp action to bounds
        u = self._clip_action(action)

        # Update internal state using synthetic model
        next_state, metrics, violation_flags = self._plant_dynamics(self.state, u)

        self.state = next_state
        self.prev_action = u

        # Build observation
        obs = self._build_observation(u, next_state, violation_flags)

        # Cost & reward
        J = self._compute_cost(metrics, violation_flags)
        reward = -J

        done = self.step_count >= self.episode_length

        info = {
            "J": J,
            "metrics": metrics,  # raw J_EM, J_P, I_HV, T_board, drift
            "violations": {
                "power": bool(violation_flags[0]),
                "current": bool(violation_flags[1]),
                "temp": bool(violation_flags[2]),
                "emi": bool(violation_flags[3]),
            },
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clip_action(self, action):
        """
        Clip a raw action to the valid bounds.
        """
        V, f, D = action
        V = np.clip(V, self.V_min, self.V_max)
        f = np.clip(f, self.f_min, self.f_max)
        D = np.clip(D, self.D_min, self.D_max)
        return np.array([V, f, D], dtype=float)

    def _plant_dynamics(self, state, u):
        """
        Synthetic nonlinear, noisy plant model.

        state = [em_state, p_state, T_board, drift]
        u     = [V_HV, f_HV, D]

        Returns:
            next_state     : updated state vector
            metrics        : dict with J_EM, J_P, I_HV, T_board, drift
            violation_flags: [power_violation, current_violation,
                              temp_violation, emi_violation]
        """
        em_state, p_state, T_board, drift = state
        V, f, D = u

        # Normalize action to [0, 1] for shaping functions
        Vn = (V - self.V_min) / (self.V_max - self.V_min + 1e-9)
        fn = (f - self.f_min) / (self.f_max - self.f_min + 1e-9)
        Dn = (D - self.D_min) / (self.D_max - self.D_min + 1e-9)

        # --------------------------------------------------------------
        # EM metric model (J_EM): multi-modal + drift + noise
        # --------------------------------------------------------------

        # Base surface: some smooth combination of sin/cos
        em_surface = (
            1.0
            + 0.4 * np.sin(2 * np.pi * Vn)
            + 0.3 * np.cos(2 * np.pi * Dn)
            + 0.2 * np.sin(2 * np.pi * fn + 0.5)
        )

        # A "good valley" near a particular region in (Vn, Dn, fn)
        valley = np.exp(-8.0 * ((Vn - 0.7) ** 2 + (Dn - 0.4) ** 2 + (fn - 0.3) ** 2))

        # Raw EM metric (before drift and noise)
        J_EM_raw = self.P_rx_off * (em_surface - 0.6 * valley)

        # Time-varying drift (slowly changing offset), scenario-dependent
        drift_next = drift + self.drift_rate * (self.rng.normal() * 0.5)
        J_EM_raw = J_EM_raw * (1.0 + 0.1 * drift_next)

        # EM measurement noise
        J_EM = J_EM_raw * (1.0 + self.noise_level_em * self.rng.normal())

        # Optional clamp to avoid negative EM (for stability in log10)
        # Here we keep it "soft" and clamp only in KPI computation.

        # --------------------------------------------------------------
        # Power model (J_P): increases with V and D, mild f effect + noise
        # --------------------------------------------------------------
        P_plasma_raw = (
            6.0 * (Vn ** 2)
            + 3.0 * Dn
            + 1.0 * fn
        )
        P_plasma = P_plasma_raw * (1.0 + self.noise_level_p * self.rng.normal())
        P_plasma = max(P_plasma, 0.0)

        # HV current model (roughly proportional to power)
        I_HV = 0.5 * P_plasma  # arbitrary scaling

        # --------------------------------------------------------------
        # Temperature dynamics (first-order ODE)
        # dT/dt = a * P_plasma - b * (T - T_ambient)
        # --------------------------------------------------------------
        a = 0.8
        b = 0.2
        dTdt = a * P_plasma - b * (T_board - self.T_ambient)
        T_board_next = T_board + self.dt * dTdt

        # --------------------------------------------------------------
        # Constraints
        # --------------------------------------------------------------
        power_violation = int(P_plasma > self.P_p_max)
        current_violation = int(I_HV > self.I_max)
        temp_violation = int(T_board_next > self.T_max)
        emi_violation = 0  # placeholder for a future EMI proxy model

        violation_flags = [
            power_violation,
            current_violation,
            temp_violation,
            emi_violation,
        ]

        # --------------------------------------------------------------
        # Update internal EM and power "states" (smoothed)
        # --------------------------------------------------------------
        em_state_next = 0.8 * em_state + 0.2 * J_EM
        p_state_next = 0.8 * p_state + 0.2 * P_plasma

        next_state = np.array(
            [em_state_next, p_state_next, T_board_next, drift_next],
            dtype=float,
        )

        metrics = {
            "J_EM": J_EM,
            "J_P": P_plasma,
            "I_HV": I_HV,
            "T_board": T_board_next,
            "drift": drift_next,
        }

        return next_state, metrics, violation_flags

    def _build_observation(self, action, state, violation_flags):
        """
        Build observation vector from current state and last action.

        Observation layout (10D):
            0: norm_J_EM  ~ em_state / P_rx_off
            1: norm_J_P   ~ p_state / P_p_max
            2: norm_T     ~ (T_board - T_ambient) / (T_max - T_ambient)
            3: Vn         ~ normalized V
            4: fn         ~ normalized f
            5: Dn         ~ normalized D
            6: power_violation (0/1)
            7: current_violation (0/1)
            8: temp_violation (0/1)
            9: emi_violation (0/1)
        """
        em_state, p_state, T_board, drift = state
        V, f, D = action
        power_violation, current_violation, temp_violation, emi_violation = violation_flags

        # Normalize EM and power roughly to [0, 1]
        norm_J_EM = em_state / (self.P_rx_off + 1e-9)
        norm_J_P = p_state / (self.P_p_max + 1e-9)

        # Normalize temperature relative to limits
        norm_T = (T_board - self.T_ambient) / (self.T_max - self.T_ambient + 1e-9)

        # Normalize action components to [0, 1]
        Vn = (V - self.V_min) / (self.V_max - self.V_min + 1e-9)
        fn = (f - self.f_min) / (self.f_max - self.f_min + 1e-9)
        Dn = (D - self.D_min) / (self.D_max - self.D_min + 1e-9)

        obs = np.array(
            [
                norm_J_EM,
                norm_J_P,
                norm_T,
                Vn,
                fn,
                Dn,
                float(power_violation),
                float(current_violation),
                float(temp_violation),
                float(emi_violation),
            ],
            dtype=float,
        )

        return obs

    def _compute_cost(
        self,
        metrics,
        violation_flags,
        w_em=0.7,
        w_p=0.3,
        lambda_viol=5.0,
    ):
        """
        Compute augmented cost J_k^aug from metrics and constraint flags.

        Base cost:
            J = w_em * (J_EM / P_rx_off) + w_p * (J_P / P_p_max)

        Augmented with constraint penalty if any violation occurs.
        """
        J_EM = metrics["J_EM"]
        J_P = metrics["J_P"]

        # Base multi-objective cost
        J_em_norm = J_EM / (self.P_rx_off + 1e-9)
        J_p_norm = J_P / (self.P_p_max + 1e-9)

        J = w_em * J_em_norm + w_p * J_p_norm

        # Constraint penalty
        any_violation = int(any(violation_flags))
        J_aug = J + lambda_viol * any_violation

        return J_aug
