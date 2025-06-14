

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta



st.title("HPB FinSage 🤖 AI-savjetnik za životne promjene ")

st.subheader("Profil & obrasci")
client = {"name": "Marko", "age": 27, "city": "Split"}

# -- Client summary as metrics --
col1, col2, col3 = st.columns(3)
col1.metric(label="👤 Ime",             value=client["name"])
col2.metric(label="🎂 Dob (godine)",    value=f"{client['age']} g")
col3.metric(label="📍 Grad",            value=client["city"])

st.markdown("---")


# Generate dummy tx for analysis
banks = ["ZABA","PBZ","ERSTE","Revolut","Split banka"]
merchants = ["Netflix","Revolut pretplata","Spotify","DM","Plodine","A1","Poslodavac Ltd"]
cats = ["Subscription","Groceries","Utilities","Salary"]
dates = pd.date_range(end=datetime.today(), periods=50, freq="7D")
tx = []
np.random.seed(1)
for d in dates:
    m = np.random.choice(merchants)
    cat = "Salary" if m=="Poslodavac Ltd" else np.random.choice(cats)
    amt = np.random.uniform(1000,2000) if cat=="Salary" else -np.random.uniform(10,300)
    b = np.random.choice(banks)
    tx.append({"date":d,"merchant":m,"category":cat,"amount":amt,"bank":b})
df = pd.DataFrame(tx)

# Client Metrics


# compute
top3       = df["merchant"].value_counts().nlargest(3).index.tolist()
worst_cat  = df.groupby("category")["amount"].sum().idxmin()
fav_bank   = df["bank"].mode()[0]
income_cnt = df[df["category"] == "Salary"].shape[0]

c1, c2 = st.columns(2)
with c1:
    st.metric("🏆 Top 3 trgovca", ", ".join(top3))
    st.metric("⚠️ Najveći rashod", worst_cat)
with c2:
    st.metric("🏦 Najomiljenija banka", fav_bank)
    st.metric("💰 Broj uplata plaće", f"{income_cnt}x")
st.markdown("---")



# Life-event detector
st.subheader("🔮 Life-Event detektor")
st.info("🔍 Porast troškova djetinjih artikala – moguća beba.")
st.info("🔍 Promjena lokacije troškova – moguća selidba.")
st.info("🔍 Neredovite uplate – moguća promjena poslodavca.")

st.markdown("---")

# AI Insights
st.subheader("🎯 AI preporuke")
st.success("Klijent troši 30% više na pretplate od sličnih dobnih skupina.")
st.warning("Moguća promjena posla – pauza u uplati plaće.")
st.info("Preporučeno razgovarati o investicijama – stabilan prihod.")

st.markdown("---")

st.button("📞 Kontaktiraj klijenta")
