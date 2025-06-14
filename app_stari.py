import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# ---------- Settings ----------
st.set_page_config(
    page_title="HPB Fincast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Data Loading & Dummy Generation ----------
@st.cache_data
def load_data():
    # Single client 'Marko'
    client = {"client_id": 2, "name": "Marko", "age": 45, "city": "Split"}
    # Simulirane transakcije kroz zadnjih 12 mjeseci
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='7D')
    banks = ["ZABA", "PBZ", "ERSTE", "Revolut", "Split banka"]

    subscriptions = ["Netflix", "Revolut pretplata", "Spotify"]
    tx = []
    np.random.seed(42)

    # Simulirane pretplate s ispravnim iznosima
    for i, merchant in enumerate(subscriptions):
        tx.append({
            "client_id": 2,
            "date": end_date - timedelta(days=(i + 1) * 10),
            "amount": -round(np.random.uniform(1, 50), 2),
            "merchant": merchant,
            "category": "Subscription",
            "bank": "Revolut" if "Revolut" in merchant else "PBZ"
        })

    # Dodatne transakcije radi grafa
    for date in dates:
        if np.random.rand() < 0.3:
            merchant = np.random.choice(["DM", "Plodine", "A1", "Poslodavac Ltd"])
            if merchant == "Poslodavac Ltd":
                amount = round(np.random.uniform(1000, 3000), 2)
                category = "Salary"
            else:
                amount = -round(np.random.uniform(10, 500), 2)
                category = "Groceries"
            bank = np.random.choice(banks)
            tx.append({
                "client_id": 2,
                "date": date,
                "amount": amount,
                "merchant": merchant,
                "category": category,
                "bank": bank
            })

    transactions = pd.DataFrame(tx)
    return client, transactions

client, tx_df = load_data()

# ---------- Header ----------
st.title("HPB Fincast ⚡️ Financijska prognoza")
col1, col2, col3, col4 = st.columns([2,2,3,3])
col1.metric("Klijent", client['name'])
col2.metric("Grad", client['city'])
current_balance = 5000.00
avg_cf = 2100.00
col3.metric("Trenutni saldo", f"{current_balance:.2f} €")
col4.metric("Prosj. mj. cash-flow", f"{avg_cf:.0f} €")

st.markdown("---")

# ---------- Cashflow Chart (Interactive) ----------
months = pd.period_range(end=datetime.today(), periods=12, freq='M').to_timestamp()
vals = np.random.normal(loc=avg_cf, scale=300, size=len(months))
monthly_cf = pd.DataFrame({'month': months, 'amount': np.round(vals, 2)})

fig = px.line(
    monthly_cf,
    x='month',
    y='amount',
    title='Mjesečni cash-flow (zadnjih 12 mjeseci)',
    markers=True,
    labels={'month':'Mjesec','amount':'Saldo (€)'}
)
fig.update_traces(line=dict(width=4), marker=dict(size=8, symbol='circle'))
fig.update_layout(transition_duration=500, xaxis=dict(tickformat="%b %Y"))
st.plotly_chart(fig, use_container_width=True)

# ---------- Forecast Cards ----------
st.subheader("⛅ Financijska vremenska prognoza za sljedeća 3 mjeseca")
forecast_months = pd.period_range(start=months[-1] + pd.offsets.MonthBegin(), periods=3, freq='M').to_timestamp()
statuses = ["☀️ Sunčano: stabilan prihod", "🌥️ Promjenjivo: mogući manjak", "⛈️ Olujno: prijeti minus"]
f_cols = st.columns(3)
for m, stat, col in zip(forecast_months, statuses, f_cols):
    with col:
        lbl = m.strftime('%B %Y')
        st.markdown(f"#### {lbl}")
        if "Sunčano" in stat:
            st.success(stat)
        elif "Promjenjivo" in stat:
            st.info(stat)
        else:
            st.error(stat)

st.markdown("---")

# ---------- Subscriptions (Expander) ----------
st.subheader("🔁 Detekcija pretplata")
with st.expander("Prikaži detalje pretplata"):
    recurring = tx_df[tx_df.merchant.str.contains("Spotify|Netflix|Revolut", case=False)]
    if recurring.empty:
        st.write("Nema aktivnih pretplata u zadnja 2 mjeseca.")
    else:
        rec_summary = recurring.groupby("merchant")["amount"].sum().reset_index()
        st.table(rec_summary)
        total = rec_summary["amount"].sum()
        st.success(f"Ukupno troškovi pretplata: {total:.2f} €")

st.markdown("---")

# ---------- All Transactions (Expander) ----------
st.subheader("💼 Sve transakcije iz svih banaka")
with st.expander("Prikaži sve transakcije"):
    df_display = tx_df.loc[:, ['date','bank','merchant','category','amount']].copy()
    df_display.columns = ['Datum','Banka','Trgovac','Kategorija','Iznos (€)']
    df_display['Datum'] = df_display['Datum'].dt.strftime('%Y-%m-%d')
    st.dataframe(df_display, use_container_width=True)

st.markdown("---")

# ---------- Call to Action ----------
col_a, col_b, col_c = st.columns([1,2,1])
with col_b:
    st.button("📞 Zatraži poziv bankara")
