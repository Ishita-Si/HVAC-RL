"""Rule-based controllers to benchmark the RL agent against."""
import numpy as np


def make_thermostat(occ_lo=21.5, occ_hi=23.5, un_lo=17.0, un_hi=27.0, gain=0.8):
    """Proportional thermostat with occupancy setback. All parameters are tunable."""
    def policy(obs, env):
        T = env.T
        lo, hi = (occ_lo, occ_hi) if env.occupied(env._hour(env.k)) else (un_lo, un_hi)
        if T < lo:
            u = np.clip(gain * (lo - T), 0, 1)
        elif T > hi:
            u = -np.clip(gain * (T - hi), 0, 1)
        else:
            u = 0.0
        return np.array([u], dtype=np.float32)
    return policy


thermostat_policy = make_thermostat()   # the original hand-picked baseline


def random_policy(obs, env):
    return env.action_space.sample()


def run_episode(env, policy, seed=None):
    """Roll out one 24 h episode. Returns (metrics dict, trajectory dict)."""
    obs, _ = env.reset(seed=seed)
    traj = {k: [] for k in ("T_in", "T_out", "P", "hour")}
    tot = dict(cost=0.0, energy_kwh=0.0, discomfort_Kh=0.0)
    done = False
    while not done:
        obs_for_policy = obs
        action = policy(obs_for_policy, env)
        obs, r, done, _, info = env.step(action)
        for k in traj:
            traj[k].append(info[k])
        for k in tot:
            tot[k] += info[k]
    return tot, {k: np.array(v) for k, v in traj.items()}
