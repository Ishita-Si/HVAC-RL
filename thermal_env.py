"""Single-zone building thermal model (1R1C) wrapped as a Gymnasium environment.

Physics (Euler step):  C * dT/dt = (T_out - T) / R + P_hvac + Q_gain
    T      indoor temperature [degC]
    T_out  outdoor temperature [degC]
    R      thermal resistance of the envelope [degC/kW]
    C      thermal capacity [kWh/degC]
    P_hvac heating (+) / cooling (-) thermal power [kW], |P| <= p_max
    Q_gain internal + solar gains [kW]

The agent controls P_hvac. Electricity use is |P| / COP.
Goal: stay in the comfort band while occupied, at minimum time-of-use cost.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ThermalZoneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, R=2.0, C=5.0, p_max=10.0, cop=3.0, dt_h=0.25,
                 comfort_occ=(21.0, 24.0), comfort_unocc=(16.0, 28.0),
                 w_comfort=0.5, out_mean=None, randomize_building=False, w_smooth=0.0, reward_scale=1.0):
        super().__init__()
        self.R0, self.C0, self.p_max, self.cop, self.dt = R, C, p_max, cop, dt_h
        self.R, self.C = R, C
        self.randomize_building = randomize_building   # domain randomisation of R, C (+-30%)
        self.comfort_occ, self.comfort_unocc = comfort_occ, comfort_unocc
        self.w_comfort = w_comfort
        self.reward_scale = reward_scale                # scales reward only; metrics in `info` are unscaled
        self.w_smooth = w_smooth                        # penalty on |a_t - a_{t-1}| (anti-chatter)
        self.fixed_out_mean = out_mean          # None -> randomised per episode
        self.n_steps = int(round(24 / dt_h))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(7,), dtype=np.float32)

    # ---------- exogenous signals ----------
    @staticmethod
    def price(hour):                      # time-of-use tariff [currency/kWh]
        if 17 <= hour < 21:
            return 0.30
        if 7 <= hour < 17 or 21 <= hour < 23:
            return 0.15
        return 0.08

    @staticmethod
    def occupied(hour):
        return 7 <= hour < 21

    def _hour(self, k):
        return (k * self.dt) % 24

    def _make_weather(self):
        mean = self.fixed_out_mean if self.fixed_out_mean is not None else self.np_random.uniform(-2, 32)
        amp = self.np_random.uniform(3, 7)
        t = np.arange(self.n_steps + 8) * self.dt
        base = mean + amp * np.sin(2 * np.pi * (t - 9) / 24)     # peak ~15:00
        noise = np.zeros_like(base)
        for i in range(1, len(noise)):                           # smooth AR(1) noise
            noise[i] = 0.95 * noise[i - 1] + self.np_random.normal(0, 0.15)
        return base + noise

    def _gains(self, hour):
        internal = 1.0 if self.occupied(hour) else 0.2
        solar = 1.5 * max(0.0, np.sin(np.pi * (hour - 7) / 10)) if 7 <= hour <= 17 else 0.0
        return internal + solar

    # ---------- gym API ----------
    def _obs(self):
        h = self._hour(self.k)
        ahead = self.t_out[self.k + 4]   # 1-hour-ahead forecast
        return np.array([
            (self.T - 22) / 10, (self.t_out[self.k] - 15) / 15, (ahead - 15) / 15,
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            self.price(h) / 0.3, float(self.occupied(h)),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_building:
            self.R = self.R0 * self.np_random.uniform(0.7, 1.3)
            self.C = self.C0 * self.np_random.uniform(0.7, 1.3)
        else:
            self.R, self.C = self.R0, self.C0
        self.t_out = self._make_weather()
        self.T = float(self.np_random.uniform(18, 26))
        self.k = 0
        self.prev_a = 0.0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
        P = a * self.p_max
        h = self._hour(self.k)
        T_out = self.t_out[self.k]

        dT = self.dt / self.C * ((T_out - self.T) / self.R + P + self._gains(h))
        self.T += dT

        lo, hi = self.comfort_occ if self.occupied(h) else self.comfort_unocc
        discomfort = max(0.0, lo - self.T, self.T - hi)          # K outside band
        energy_kwh = abs(P) / self.cop * self.dt
        cost = energy_kwh * self.price(h)
        reward = -(self.w_comfort * discomfort * self.dt + cost + self.w_smooth * abs(a - self.prev_a))
        self.prev_a = a
        reward *= self.reward_scale

        self.k += 1
        terminated = self.k >= self.n_steps
        info = dict(T_in=self.T, T_out=T_out, P=P, hour=h, cost=cost,
                    energy_kwh=energy_kwh, discomfort_Kh=discomfort * self.dt)
        obs = self._obs() if not terminated else np.zeros(7, dtype=np.float32)
        return obs, reward, terminated, False, info
