"""Shared helpers: batch evaluation, thermostat tuning, Pareto front, paired statistics.
No torch import here, so the thermostat side can be run anywhere."""
import itertools
import numpy as np
import pandas as pd
from baselines import make_thermostat, run_episode
from thermal_env import ThermalZoneEnv

TUNE_SEEDS = list(range(0, 60))            # used ONLY to pick thermostat parameters
TEST_SEEDS = list(range(10_000, 10_100))   # used ONLY for reported results

GRID = dict(occ_lo=[21.0, 21.5, 22.0], occ_hi=[23.0, 23.5, 24.0],
            un_lo=[18.0, 19.0, 20.0, 21.0], un_hi=[24.0, 25.0, 26.0, 27.0], gain=[1.0, 2.0, 4.0])


def rollout_df(policy, env, seeds):
    """Per-episode metrics for `policy` on the given weather seeds."""
    return pd.DataFrame([run_episode(env, policy, seed=s)[0] for s in seeds])


def objective(df, w_comfort):
    """Same quantity the RL agent minimises (per day): cost + w * discomfort K-h."""
    return df["cost"] + w_comfort * df["discomfort_Kh"]


def tune_grid(env, seeds=TUNE_SEEDS, grid=GRID):
    rows = []
    for combo in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), combo))
        if p["occ_lo"] >= p["occ_hi"]:
            continue
        m = rollout_df(make_thermostat(**p), env, seeds).mean()
        rows.append({**p, "cost": m.cost, "energy_kwh": m.energy_kwh, "discomfort_Kh": m.discomfort_Kh})
    return pd.DataFrame(rows)


def best_params(grid_df, w_comfort):
    score = grid_df["cost"] + w_comfort * grid_df["discomfort_Kh"]
    row = grid_df.loc[score.idxmin()]
    return {k: float(row[k]) for k in GRID}


def pareto_front(df, x="cost", y="discomfort_Kh"):
    """Non-dominated rows (lower is better on both axes)."""
    d = df.sort_values([x, y]).reset_index(drop=True)
    keep, best_y = [], np.inf
    for i, r in d.iterrows():
        if r[y] < best_y - 1e-9:
            keep.append(i); best_y = r[y]
    return d.loc[keep]


def paired_stats(df_a, df_b, w_comfort, n_boot=5000, seed=0):
    """A vs B on identical weather. Negative diff = A better (lower objective)."""
    d = (objective(df_a, w_comfort) - objective(df_b, w_comfort)).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean_diff=d.mean(), ci95_low=lo, ci95_high=hi,
                a_wins_pct=100 * float((d < 0).mean()),
                rel_improvement_pct=100 * -d.mean() / objective(df_b, w_comfort).mean())
