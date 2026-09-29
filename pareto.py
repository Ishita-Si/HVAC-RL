"""Fair comparison: PPO vs a *tuned* thermostat, across comfort weights.

    python pareto.py                       # full run: trains 1 PPO per weight (slow, ~10-30 min each)
    python pareto.py --steps 300000        # quicker
    python pareto.py --skip_train          # reuse models/w*.zip from an earlier run
    python pareto.py --no_ppo              # thermostat frontier only (fast sanity check)

Outputs: pareto.png, pareto_results.csv, paired_stats.csv, tuned_params.json, thermostat_grid.csv
Protocol: thermostat parameters are tuned on TUNE_SEEDS; every number reported is on unseen TEST_SEEDS.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from thermal_env import ThermalZoneEnv
from baselines import make_thermostat, thermostat_policy
from experiments import (GRID, TEST_SEEDS, rollout_df, objective, tune_grid,
                         best_params, pareto_front, paired_stats)


def summarize(df, policy, w):
    m = df.mean()
    return dict(policy=policy, w_comfort=w, cost=m.cost, energy_kwh=m.energy_kwh,
                discomfort_Kh=m.discomfort_Kh, objective=float(objective(df, w).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--w_smooth", type=float, default=0.0)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--no_ppo", action="store_true")
    a = ap.parse_args()

    env = ThermalZoneEnv()
    print("Tuning thermostat grid on tuning seeds ...")
    grid = tune_grid(env)
    grid.to_csv("thermostat_grid.csv", index=False)

    # thermostat frontier, re-evaluated on unseen test seeds
    pts = []
    for _, r in pareto_front(grid).iterrows():
        m = rollout_df(make_thermostat(**{k: float(r[k]) for k in GRID}), env, TEST_SEEDS).mean()
        pts.append(dict(cost=m.cost, discomfort_Kh=m.discomfort_Kh))
    front_test = pareto_front(pd.DataFrame(pts))

    rows, stats, tuned = [], [], {}
    default_df = rollout_df(thermostat_policy, env, TEST_SEEDS)
    for w in a.weights:
        p = best_params(grid, w)
        tuned[str(w)] = p
        th_df = rollout_df(make_thermostat(**p), env, TEST_SEEDS)
        rows.append(summarize(default_df, "Default thermostat", w))
        rows.append(summarize(th_df, "Tuned thermostat", w))
        if a.no_ppo:
            continue
        out = f"models/w{w}"
        if not a.skip_train:
            subprocess.run([sys.executable, "train.py", "--steps", str(a.steps), "--w_comfort", str(w),
                            "--w_smooth", str(a.w_smooth), "--out", out], check=True)
        from stable_baselines3 import PPO
        best = f"{out}_best/best_model.zip"
        model = PPO.load(best if os.path.exists(best) else out + ".zip")
        ppo_df = rollout_df(lambda obs, e: model.predict(obs, deterministic=True)[0], env, TEST_SEEDS)
        rows.append(summarize(ppo_df, "PPO", w))
        s = paired_stats(ppo_df, th_df, w)
        s["w_comfort"] = w
        s["verdict"] = ("PPO better" if s["ci95_high"] < 0 else
                        "Thermostat better" if s["ci95_low"] > 0 else "No significant difference")
        stats.append(s)

    res = pd.DataFrame(rows)
    res.round(3).to_csv("pareto_results.csv", index=False)
    json.dump(tuned, open("tuned_params.json", "w"), indent=2)
    print("\n" + res.round(3).to_string(index=False))
    if stats:
        st = pd.DataFrame(stats)[["w_comfort", "mean_diff", "ci95_low", "ci95_high",
                                  "a_wins_pct", "rel_improvement_pct", "verdict"]]
        st.round(3).to_csv("paired_stats.csv", index=False)
        print("\nPPO vs TUNED thermostat (paired, same weather; diff<0 => PPO better):")
        print(st.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(grid["cost"], grid["discomfort_Kh"], s=8, color="lightgray", label="Thermostat settings (grid)")
    ax.plot(front_test["cost"], front_test["discomfort_Kh"], "o-", color="tab:blue", label="Thermostat Pareto front (test)")
    d = res[res.policy == "Default thermostat"].iloc[0]
    ax.scatter(d.cost, d.discomfort_Kh, marker="x", s=80, color="k", label="Default thermostat", zorder=5)
    ppo = res[res.policy == "PPO"]
    if len(ppo):
        ax.plot(ppo["cost"], ppo["discomfort_Kh"], "s--", color="tab:red", label="PPO (one per weight)")
        for _, r in ppo.iterrows():
            ax.annotate(f"w={r.w_comfort:g}", (r.cost, r.discomfort_Kh), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Electricity cost per day"); ax.set_ylabel("Discomfort (K·h per day)")
    ax.set_title("Lower-left is better"); ax.legend(fontsize=8); ax.set_ylim(bottom=0)
    plt.tight_layout(); plt.savefig("pareto.png", dpi=150)
    print("\nsaved pareto.png, pareto_results.csv, paired_stats.csv, tuned_params.json, thermostat_grid.csv")


if __name__ == "__main__":
    main()
