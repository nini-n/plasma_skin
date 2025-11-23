# controllers/extremum_seeking.py

import numpy as np

class ExtremumSeekingController:
    """
    Simple stochastic hill-climbing / extremum-seeking controller.
    - Keeps track of best known action (best_u, best_J).
    - At each step proposes a small random move in action space,
      and keeps it only if the cost improves.
    - Uses reward/info from previous step (reward = -J_aug).
    """

    def __init__(
        self,
        V_min, V_max,
        f_min, f_max,
        D_min, D_max,
        seed=None,
        initial_step_scale=0.1,
        min_step_scale=0.01,
        step_decay=0.999,
    ):
        self.V_min, self.V_max = V_min, V_max
        self.f_min, self.f_max = f_min, f_max
        self.D_min, self.D_max = D_min, D_max

        self.rng = np.random.default_rng(seed)

        # Step size parameters (relative to action range)
        self.initial_step_scale = initial_step_scale
        self.min_step_scale = min_step_scale
        self.step_decay = step_decay

        # Internal state
        self.best_u = None
        self.best_J = None
        self.candidate_u = None
        self.step_scale = initial_step_scale

        # We'll store last cost we evaluated for candidate
        self.last_J = None

    def reset(self):
        """
        Reset controller internal state.
        Initial best_u is the mid-range safe action.
        """
        V0 = 0.5 * (self.V_min + self.V_max)
        f0 = 0.5 * (self.f_min + self.f_max)
        D0 = 0.3

        self.best_u = np.array([V0, f0, D0], dtype=float)
        self.best_J = None
        self.candidate_u = None
        self.step_scale = self.initial_step_scale
        self.last_J = None

        # First action is just the best known
        return self.best_u.copy()

    def compute_action(self, obs, reward=None, info=None):
        """
        Compute next action.

        Convention in loop:
        - At step k, we call compute_action(obs_k, reward_{k-1}, info_{k-1}).
        - So reward/info correspond to the cost of the *previous* action.
        """
        # 1) Use previous reward to update best_u / best_J
        if reward is not None and info is not None:
            # reward = -J_aug
            J = -reward

            # Eğer daha önce hiç cost görmediysek, bunu best_J yap
            if self.best_J is None:
                self.best_J = J
            else:
                # Bir önceki adımda candidate_u uygulandıysa:
                if self.candidate_u is not None:
                    # Eğer yeni cost daha iyiyse (daha küçükse) -> kabul et
                    if J < self.best_J:
                        self.best_J = J
                        self.best_u = self.candidate_u.copy()
                        # İyileşme varsa step_scale'i biraz büyütebiliriz (opsiyonel)
                        # self.step_scale = min(self.step_scale / self.step_decay, self.initial_step_scale)
                    else:
                        # Cost kötüleşti, candidate'i reddet
                        # İsteğe bağlı: step_scale'i biraz küçült
                        self.step_scale = max(self.step_scale * self.step_decay, self.min_step_scale)

            self.last_J = J

        # 2) Yeni candidate aksiyon üret (best_u etrafında küçük random adım)
        if self.best_u is None:
            # Emniyet için; normalde reset sırasında set ediliyor
            self.reset()

        # Rastgele yön (3 boyutta)
        direction = self.rng.normal(size=3)
        direction /= (np.linalg.norm(direction) + 1e-9)

        # Her eksendeki aralık
        range_V = self.V_max - self.V_min
        range_f = self.f_max - self.f_min
        range_D = self.D_max - self.D_min

        # Step vektörü
        step_vec = np.array([
            direction[0] * range_V,
            direction[1] * range_f,
            direction[2] * range_D,
        ])

        # Küçük adım
        candidate = self.best_u + self.step_scale * step_vec

        # Sınırla
        candidate[0] = np.clip(candidate[0], self.V_min, self.V_max)
        candidate[1] = np.clip(candidate[1], self.f_min, self.f_max)
        candidate[2] = np.clip(candidate[2], self.D_min, self.D_max)

        self.candidate_u = candidate

        # Şu adımda environment'a değerlendirmesi için candidate aksiyonu gönderiyoruz
        return self.candidate_u.copy()
