"""Streamlit demo:  streamlit run app.py"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from thermal_env import ThermalZoneEnv
from baselines import thermostat_policy, run_episode

st.set_page_config(page_title="RL Thermal Digital Twin", layout="wide")
st.title("RL-controlled building thermal digital twin")
st.caption("1R1C physics model of a room; PPO agent vs rule-based thermostat.")

with st.sidebar:
    R = st.slider("Envelope resistance R (°C/kW)", 0.8, 4.0, 2.0, 0.1)
    C = st.slider("Thermal capacity C (kWh/°C)", 2.0, 10.0, 5.0, 0.5)
    out_mean = st.slider("Mean outdoor temp (°C)", -5, 35, 5)
    seed = st.number_input("Weather seed", 0, 9999, 7)

env = ThermalZoneEnv(R=R, C=C, out_mean=float(out_mean))
policies = {"Thermostat": thermostat_policy}
model_path = next((p for p in ("models/best/best_model.zip", "models/ppo_thermal.zip") if os.path.exists(p)), None)
if model_path:
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    policies["PPO agent"] = lambda obs, e: model.predict(obs, deterministic=True)[0]
else:
    st.warning("No trained model found - run `python train.py` first. Showing the thermostat only.")

cols = st.columns(len(policies))
temp_df, pow_df = {}, {}
for col, (name, pol) in zip(cols, policies.items()):
    m, tr = run_episode(env, pol, seed=int(seed))
    with col:
        st.subheader(name)
        st.metric("Cost", f"{m['cost']:.2f}")
        st.metric("Energy (kWh)", f"{m['energy_kwh']:.1f}")
        st.metric("Discomfort (K·h)", f"{m['discomfort_Kh']:.2f}")
    temp_df[name] = tr["T_in"]; pow_df[name] = tr["P"]
hours = tr["hour"]
temp_df["Outdoor"] = tr["T_out"]
st.subheader("Indoor temperature (comfort band 21-24 °C while occupied, 07-21 h)")
st.line_chart(pd.DataFrame(temp_df, index=hours))
st.subheader("HVAC thermal power (kW, + heating / - cooling)")
st.line_chart(pd.DataFrame(pow_df, index=hours))
