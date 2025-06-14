import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# ---------- Settings ----------
st.set_page_config(
    page_title="HPB Fincast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Style ----------
st.markdown("""
    <style>
        .metric-label > div {
            font-size: 18px;
        }
        .card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Data Loading & Dummy Generation ----------
@st.cache_data
def load_data():
    client = {"client_id": 2, "name": "Marko", "age": 27, "city": "Zagreb"}
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='7D')
    banks = ["ZABA", "PBZ", "ERSTE", "Revolut", "Split banka"]

    subscriptions = ["Netflix", "Revolut pretplata", "Spotify"]
    tx = []
    np.random.seed(42)

    for i, merchant in enumerate(subscriptions):
        tx.append({
            "client_id": 2,
            "date": end_date - timedelta(days=(i + 1) * 10),
            "amount": -round(np.random.uniform(1, 50), 2),
            "merchant": merchant,
            "category": "Subscription",
            "bank": "Revolut" if "Revolut" in merchant else "PBZ"
        })

    for date in dates:
        if np.random.rand() < 0.3:
            merchant = np.random.choice(["DM", "Plodine", "Konzum", "Poslodavac Ltd"])
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
    title='📈 Mjesečni priljev (zadnjih 12 mjeseci)',
    markers=True,
    labels={'month':'Mjesec','amount':'Saldo (€)'}
)
fig.update_traces(line=dict(width=4), marker=dict(size=8, symbol='circle'))
fig.update_layout(transition_duration=500, xaxis=dict(tickformat="%b %Y"))
st.plotly_chart(fig, use_container_width=True)

# ---------- Forecast Cards ----------
st.subheader("⛅ Financijska vremenska prognoza")
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

# ---------- All Transactions (Expander) ----------
st.subheader("💼 Sve transakcije iz svih banaka")
with st.expander("Prikaži sve transakcije"):
    # Prepare display DataFrame
    df_display = tx_df.loc[:, ['date','bank','merchant','category','amount']].copy()
    df_display.columns = ['Datum','Banka','Trgovac','Kategorija','Iznos (€)']
    df_display['Datum'] = df_display['Datum'].dt.strftime('%Y-%m-%d')

    # Define a color map for each bank
    bank_colors = {
        "ZABA":       "#D1E8E2",  # mint
        "PBZ":        "#FAD4D4",  # light coral
        "ERSTE":      "#D4EBF2",  # sky
        "Revolut":    "#E8DFF5",  # lavender
        "Split banka":"#FFF3CF"   # light yellow
    }

    # Styler function to apply background based on bank
    def color_bank(val):
        return f"background-color: {bank_colors.get(val, 'white')}"
    
    # Apply styling only to the 'Banka' column
    styled = df_display.style.applymap(color_bank, subset=['Banka'])

    # Show styled DataFrame
    st.dataframe(styled, use_container_width=True)

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

# ---------- Analytics Insights ----------
st.subheader("🧠 Klijent u brojkama")

# Compute stats
merchant_counts  = tx_df['merchant'].value_counts()
category_sums    = tx_df.groupby('category')['amount'].sum().abs()
bank_counts      = tx_df['bank'].value_counts()
num_incomes      = tx_df[tx_df['category']=="Salary"].shape[0]
net_balance      = tx_df['amount'].sum()

# Layout 2×2
r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

# 1) Merchant distribution pie
with r1c1:
    fig1 = px.pie(
        names=merchant_counts.index,
        values=merchant_counts.values,
        title="Vodeći trgovci(broj tx)",
        hole=0.4
    )
    st.plotly_chart(fig1, use_container_width=True)

# 2) Spend by category bar
with r2c1:
    fig2 = px.bar(
        x=category_sums.index,
        y=category_sums.values,
        title="Potrošnja po kategorijama",
        labels={'x':'Kategorija','y':'Iznos (€)'}
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3) Bank usage pie
with r1c2:
    fig3 = px.pie(
        names=bank_counts.index,
        values=bank_counts.values,
        title="Transakcije po banci"
    )
    st.plotly_chart(fig3, use_container_width=True)

# 4) Key metrics panel
with r2c2:
    st.markdown("### 📊 Ostale metrike")
    st.metric("Uplate plaće (x)", f"{num_incomes}")
    st.metric("Ukupan net saldo", f"{net_balance:.2f} €")

st.markdown("---")



# ---------- Top Trošak po Mjesecu ----------
st.subheader("📆 Najveći pojedinačni trošak mjesečno")
tx_df['Month'] = tx_df['date'].dt.to_period('M')
maxi = tx_df.loc[tx_df.groupby('Month')['amount'].idxmin()]
st.dataframe(maxi[['Month', 'date', 'merchant', 'amount']], use_container_width=True)

st.markdown("---")

# ---------- Anomalije ----------
st.subheader("🚨 Neobične potrošnje")
tx_df['abs'] = tx_df['amount'].abs()
threshold = tx_df['abs'].mean() + 2 * tx_df['abs'].std()
anomalies = tx_df[tx_df['abs'] > threshold]
if not anomalies.empty:
    st.error("⚠️ Otkrivene transakcije koje znatno odstupaju od prosjeka:")
    st.dataframe(anomalies[['date', 'merchant', 'amount']], use_container_width=True)
else:
    st.success("Nema značajnih anomalija.")
    
st.markdown("---")

# ---------- Usporedba s vršnjacima ----------
st.subheader("👥 Usporedba s prosjekom u tvojoj dobnoj skupini")

# Fiktivne prosječne vrijednosti za dobnu skupinu 40–50 god.
avg_peers_cf = 1600.0
avg_peers_subs = 200.0
avg_peers_tx_count = 120

user_subs_total = tx_df[tx_df['category'] == "Subscription"]['amount'].sum()
user_tx_count = len(tx_df)

colp1, colp2 = st.columns(2)
with colp1:
    st.metric("Mj. cash-flow", f"{avg_cf:.2f} €", delta=f"{avg_cf - avg_peers_cf:.2f} €")
    st.metric("Broj transakcija", f"{user_tx_count}", delta=f"{user_tx_count - avg_peers_tx_count}")
with colp2:
    st.metric("Trošak pretplata", f"{user_subs_total:.2f} €", delta=f"{user_subs_total - avg_peers_subs:.2f} €")

if avg_cf > avg_peers_cf:
    st.success("✅ Tvoj prihod je veći od prosjeka tvoje dobne skupine.")
else:
    st.warning("⚠️ Tvoj mjesečni cash-flow je ispod prosjeka vršnjaka.")

st.markdown("---")

# ---------- Smart Insight ----------
st.info("📊 Insight: Klijent Marko troši 28% više na pretplate od prosjeka populacije iste dobne skupine.")

st.markdown("---")

# ---------- Cilj štednje ----------
st.subheader("🎯 Postavi financijski cilj")
goal = st.number_input("Unesi iznos koji želiš uštedjeti (€):", min_value=0)
months_left = st.slider("U koliko mjeseci želiš dostići cilj?", 1, 24, 12)

if goal > 0:
    monthly_save = goal / months_left
    st.info(f"Ako želiš doseći {goal} € za {months_left} mjeseci, odvajaj **{monthly_save:.2f} €** mjesečno.")


st.markdown("---")




# ---------- AI Savjetnik ----------  
st.subheader("💡 Savjetnik za pametnu štednju")
tipovi = []

if category_sums.get("Subscription", 0) > 300:
    tipovi.append("Razmisli o otkazivanju barem jedne pretplate koju ne koristiš aktivno.")

if net_balance < 0:
    tipovi.append("Povećaj mjesečne prihode (npr. freelance) kako bi izbjegao minus.")

if bank_counts['PBZ'] > bank_counts.mean():
    tipovi.append("Pregledaj provizije PBZ-a – možda je vrijeme za konsolidaciju računa.")

if not tipovi:
    st.success("🎉 Tvoj financijski profil izgleda stabilno!")
else:
    for t in tipovi:
        st.warning(f"👉 {t}")

    
st.markdown("---")


# ---------- What-if Insights ----------
st.subheader("🔮 Što ako...")
st.markdown("Ako korisnik smanji pretplate za 30%, ušteda na godišnjoj razini: **182 €**")
st.markdown("Ako prebaci primanja iz PBZ u HPB, dodatni bonus: **50 €** za aktivaciju računa")

# ---------- Simulacija kredita ----------
st.subheader("💰 Simulacija kredita")

# 1. Vrste kredita s tipičnim kamatama
credit_types = {
    "Stambeni kredit": 0.025,
    "Potrošački kredit": 0.065,
    "Kredit za kupnju vozila": 0.045,
    "Učenički/studentski kredit": 0.03,
    "Prekoračenje po tekućem računu": 0.08,
    "Lombardni kredit": 0.04,
    "Gotovinski kredit": 0.07
}

# 2. Izbor vrste kredita
credit_choice = st.selectbox("Odaberi vrstu kredita:", list(credit_types.keys()))
interest_rate = credit_types[credit_choice]

# 3. Iznos i trajanje
loan_col1, loan_col2 = st.columns(2)
with loan_col1:
    loan_amount = st.number_input("Iznos kredita (€):", min_value=1000, max_value=100000, value=10000, step=1000)
with loan_col2:
    loan_years = st.slider("Trajanje otplate (godina):", min_value=1, max_value=30, value=10)

monthly_rate = interest_rate / 12
months = loan_years * 12

# 4. Izračuni
if loan_amount:
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    total_payment = monthly_payment * months
    interest_cost = total_payment - loan_amount

    st.markdown(f"**Kamatna stopa za '{credit_choice}':** {interest_rate * 100:.2f}% godišnje")
    
    col_rate, col_total, col_interest = st.columns(3)
    with col_rate:
        st.info(f"📅 Mjesečna rata:\n**{monthly_payment:.2f} €**")
    with col_total:
        st.success(f"💸 Ukupno za otplatu:\n**{total_payment:.2f} €**")
    with col_interest:
        st.warning(f"📈 Ukupna kamata:\n**{interest_cost:.2f} €**")
# ---------- Kreditna sposobnost ----------
st.markdown("---")
st.subheader("🧮 Procjena kreditne sposobnosti")

# Pretpostavimo da imaš već definiran avg_cf (mjesečni cash-flow)
max_affordable_rate = avg_cf * 0.4

st.info(f"📊 Preporučena mjesečna rata (40% cash-flowa): **{max_affordable_rate:.2f} €**")

if monthly_payment > max_affordable_rate:
    st.error(f"⚠️ Odabrana rata od **{monthly_payment:.2f} €** prelazi preporučeni maksimum ({max_affordable_rate:.2f} €). Razmisli o manjem iznosu ili duljoj otplati.")
else:
    st.success(f"✅ Rata od **{monthly_payment:.2f} €** je unutar preporučenog maksimuma ({max_affordable_rate:.2f} €).")

st.markdown("---")

# ---------- CTA dugme ----------
st.markdown("### 🤝 Želiš osobnu ponudu?")
st.button("📞 Zatraži sastanak s HPB bankarom")







