import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from contextlib import redirect_stdout

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environments.portfolio_env import PortfolioEnv

@st.cache_data(show_spinner=False)
def simulate_rl(initial_amount: float, months: int):
    """Run a trained PPO policy in a normalized VecEnv and return portfolio values + render logs."""
    # 1) build & load VecNormalize
    vec_env = make_vec_env(
        lambda: PortfolioEnv(initial_cash=initial_amount, T=months),
        n_envs=1
    )
    vec_norm = VecNormalize.load("vec_normalize2.pkl", vec_env)
    vec_norm.training = False
    vec_norm.norm_reward = False

    # 2) load the policy
    model = PPO.load("ppo_portfolio_model2.zip", env=vec_norm)

    # 3) unwrap down to the raw Gym env
    venv = vec_norm.venv if hasattr(vec_norm, "venv") else vec_norm
    inner = venv.envs[0]
    raw_env = inner.env if isinstance(inner, Monitor) else inner

    # 4) simulate, capturing render() output
    obs, _ = raw_env.reset()
    values = []
    render_logs = []

    for _ in range(months):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = raw_env.step(action)

        # capture render() stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            raw_env.render()
        render_logs.append(buf.getvalue())

        # compute portfolio value
        cash  = raw_env.state_obj.total_cash
        bonds = raw_env.state_obj.unrealized_bonds()
        # ignore loans part for now
        values.append(cash + bonds)

        if terminated:
            break

    # pad in case of early done
    if len(values) < months:
        last_val = values[-1]
        values += [last_val] * (months - len(values))
        render_logs += [render_logs[-1]] * (months - len(render_logs))

    return values, render_logs

@st.cache_data(show_spinner=False)
def simulate_one_over_n(initial_amount: float, months: int):
    """Classic 1/N (equally‐weighted bonds, no credit/loans) baseline."""
    env = PortfolioEnv(initial_cash=initial_amount, T=months)
    obs, _ = env.reset()
    values = []
    one_over_n = np.ones(9) / 9.0
    for _ in range(months):
        total_cash = env.state_obj.total_cash
        reserve    = 0.20 * total_cash
        investable = max(total_cash - reserve, 0.0)
        bond_act   = one_over_n * investable
        action     = np.concatenate([bond_act, np.zeros(3), np.zeros(3)])
        obs, reward, terminated, truncated, info = env.step(action)
        cash  = env.state_obj.total_cash
        bonds = env.state_obj.unrealized_bonds()
        values.append(cash + bonds)
        if terminated:
            break
    if len(values) < months:
        values += [values[-1]] * (months - len(values))
    return values

@st.cache_data(show_spinner=False)
def simulate_random(initial_amount: float, months: int):
    """Random bond‐only allocation; no credits or loans."""
    env = PortfolioEnv(initial_cash=initial_amount, T=months)
    obs, _ = env.reset()
    values = []
    for _ in range(months):
        total_cash = env.state_obj.total_cash
        reserve    = 0.20 * total_cash
        investable = max(total_cash - reserve, 0.0)
        w = np.random.rand(9)
        w /= (w.sum() + 1e-8)
        bond_act = w * investable
        action = np.concatenate([bond_act, np.zeros(3), np.zeros(3)])
        obs, reward, terminated, truncated, info = env.step(action)
        cash  = env.state_obj.total_cash
        bonds = env.state_obj.unrealized_bonds()
        values.append(cash + bonds)
        if terminated:
            break
    if len(values) < months:
        values += [values[-1]] * (months - len(values))
    return values


st.title("HPB Investor AI 📈 Simulacija ulaganja")
st.markdown("Isprobaj AI strategiju nasuprot klasičnom 1/N i Random ulaganju kroz 12 mjeseci!")

inv = st.number_input(
    "Početni iznos za ulaganje (€):",
    min_value=10.0, max_value=10.0,
    value=10.0,
    disabled=True,
)
months = 30

with st.spinner("Pokrećem AI simulaciju…"):
    ai_vals, ai_logs = simulate_rl(inv, months)

with st.spinner("Pokrećem 1/N simulaciju…"):
    n_vals  = simulate_one_over_n(inv, months)

with st.spinner("Pokrećem Random simulaciju…"):
    r_vals  = simulate_random(inv, months)

# build DataFrame & plot
df = pd.DataFrame({
    "Mjesec":               [f"M{m+1}" for m in range(months)],
    "AI strategija (PPO)":  np.round(ai_vals, 2),
    "1/N strategija":       np.round(n_vals,  2),
    "Random strategija":    np.round(r_vals,  2),
})
fig = px.line(
    df,
    x="Mjesec",
    y=["AI strategija (PPO)", "1/N strategija", "Random strategija"],
    labels={"value":"Portfelj (€)", "variable":"Strategija"},
    title="Usporedba rasta portfelja"
)
st.plotly_chart(fig, use_container_width=True)

# metrics
ai_final = df["AI strategija (PPO)"].iloc[-1]
n_final  = df["1/N strategija"].iloc[-1]
r_final  = df["Random strategija"].iloc[-1]
c1, c2, c3 = st.columns(3)
postotak=(ai_final - n_final)/ai_final*100
c1.metric("AI krajnja vrijednost",    f"{ai_final:.2f} €", delta=f"{postotak:.2f} %")
c2.metric("1/N krajnja vrijednost",   f"{n_final:.2f} €")
c3.metric("Random krajnja vrijednost",f"{r_final:.2f} €")

st.markdown("---")

# show raw_env.render() logs in expanders
st.markdown("## Detaljni logovi renderiranja (AI strategija)")
for i, log in enumerate(ai_logs):
    # 1. split into individual lines
    lines = log.splitlines()
    # 2. filter out any line that is just the dash placeholder
    cleaned = [ln for ln in lines if '—' not in ln]
    # 3. only display months with real content
    if not cleaned:
        continue

    with st.expander(f"Mjesec {i+1}"):
        # re-join the “good” lines and show them
        st.text("\n".join(cleaned))

if st.button("📞 Zatraži sastanak s bankarom"):
    st.success("Vaš zahtjev je zabilježen, kontaktirat će vas bankar!")