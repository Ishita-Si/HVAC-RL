"""Train a PPO agent on the thermal-zone digital twin.

    python train.py --steps 1000000
"""
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from thermal_env import ThermalZoneEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--randomize", action="store_true", help="randomise building R, C each episode")
    ap.add_argument("--w_comfort", type=float, default=0.5)
    ap.add_argument("--w_smooth", type=float, default=0.0)
    ap.add_argument("--reward_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/ppo_thermal")
    a = ap.parse_args()

    kw = dict(randomize_building=a.randomize, w_comfort=a.w_comfort, w_smooth=a.w_smooth,
              reward_scale=a.reward_scale)
    train_env = make_vec_env(ThermalZoneEnv, n_envs=8, seed=a.seed, env_kwargs=kw)
    eval_env = make_vec_env(ThermalZoneEnv, n_envs=1, seed=a.seed + 100, env_kwargs=kw)

    model = PPO("MlpPolicy", train_env, seed=a.seed, verbose=1,
                n_steps=192, batch_size=256, gamma=0.995, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.0,
                policy_kwargs=dict(net_arch=[128, 128]), tensorboard_log="runs")
    cb = EvalCallback(eval_env, best_model_save_path=a.out + "_best", eval_freq=10_000,
                      n_eval_episodes=20, deterministic=True)
    model.learn(total_timesteps=a.steps, callback=cb, tb_log_name="ppo")
    model.save(a.out)
    print(f"saved {a.out}.zip  (best checkpoint in {a.out}_best/best_model.zip)")


if __name__ == "__main__":
    main()
