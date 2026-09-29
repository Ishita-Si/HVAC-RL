"""Compare PPO vs thermostat vs random over many unseen episodes and plot one day.

    python evaluate.py --model models/best/best_model.zip --episodes 100
    python evaluate.py --model ... --R 1.2 --C 3.5     # test on a different building
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from thermal_env import ThermalZoneEnv
import json, os
from baselines import thermostat_policy, random_policy, run_episode, make_thermostat
from experiments import rollout_df, paired_stats, objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best/best_model.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--w_comfort", type=float, default=0.5, help="weight the model was trained with")
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--C", type=float, default=5.0)
    ap.add_argument("--out_mean", type=float, default=None, help="fix outdoor mean temp (degC)")
    a = ap.parse_args()

    env = ThermalZoneEnv(R=a.R, C=a.C, out_mean=a.out_mean)
    model = PPO.load(a.model)
    ppo_policy = lambda obs, e: model.predict(obs, deterministic=True)[0]
    tuned = json.load(open("tuned_params.json")).get(str(a.w_comfort)) if os.path.exists("tuned_params.json") else None
    if tuned is None:
        print("NOTE: run pareto.py first to get a tuned thermostat; using the default one.")
    policies = {"PPO": ppo_policy,
                "Tuned thermostat": make_thermostat(**tuned) if tuned else thermostat_policy,
                "Default thermostat": thermostat_policy, "Random": random_policy}

    rows, dfs = [], {}
    for name, pol in policies.items():
        df = rollout_df(pol, env, [10_000 + s for s in range(a.episodes)])   # same seeds for all
        dfs[name] = df
        rows.append({"policy": name, **df.mean().round(3).to_dict(),
                     "objective": round(float(objective(df, a.w_comfort).mean()), 3)})
    table = pd.DataFrame(rows).set_index("policy")
    print(table.to_string())
    table.to_csv("results.csv")
    st = paired_stats(dfs["PPO"], dfs["Tuned thermostat"], a.w_comfort)
    print("\nPPO vs tuned thermostat (paired; diff<0 => PPO better):",
          {k: round(float(v), 3) for k, v in st.items()})

    # one example day
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for name, pol in policies.items():
        if name in ("Random", "Default thermostat"):
            continue
        _, tr = run_episode(env, pol, seed=10_007)
        ax[0].plot(tr["hour"], tr["T_in"], label=name)
        ax[1].plot(tr["hour"], tr["P"], label=name)
    ax[0].plot(tr["hour"], tr["T_out"], "--", color="gray", label="Outdoor")
    ax[0].axhspan(21, 24, alpha=0.12, color="green")
    ax[0].set_ylabel("Temperature (°C)"); ax[0].legend()
    ax[1].set_ylabel("HVAC power (kW)")
    ax[2].step(tr["hour"], [ThermalZoneEnv.price(h) for h in tr["hour"]], color="k")
    ax[2].set_ylabel("Tariff"); ax[2].set_xlabel("Hour of day")
    plt.tight_layout(); plt.savefig("example_day.png", dpi=150)
    print("saved results.csv, example_day.png")


if __name__ == "__main__":
    main()
