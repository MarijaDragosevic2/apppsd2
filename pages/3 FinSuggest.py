import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# import your class
from final_embedding import EmbeddingRecommender

WEIGHTS_PATH     = "best_embedding_full.pt"
DEFAULT_CLUSTERS = 5

@st.cache_data(show_spinner=False)
def load_data():
    clients = pd.read_csv("KLIJENTI_CLEANED.csv", low_memory=False)
    prods   = pd.read_csv("PROIZVODI_CLEANED.csv", low_memory=False)
    return clients, prods

@st.cache_resource(show_spinner=False)
def load_model(clients_df, produkti_df, weights_path):
    model = EmbeddingRecommender(
        client_df=clients_df,
        produkti_df=produkti_df,
        embedding_dim=32,
        alpha_pop=0.5,
        cutoff_date="2025-04-15",
    )
    state = torch.load(weights_path, map_location=model.device)
    model.load_state_dict(state)
    model.to(model.device)
    model.eval()
    return model

@st.cache_data(show_spinner=False)
def compute_embeddings(_model):
    embs = _model.encode_clients().detach().cpu().numpy()
    return embs, _model.client_ids

def run_clustering(embs, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels = km.fit_predict(embs)
    return labels, km.cluster_centers_

# --- Streamlit UI ---
st.set_page_config(page_title="🔮 HPB Recommender", layout="wide")
st.title("HPB FinSuggest 🔮 Preporuka proizvoda i analiza klustera")

# 1) load
clients_df, prods_df = load_data()
model               = load_model(clients_df, prods_df, WEIGHTS_PATH)
embs, client_ids    = compute_embeddings(model)

# 2) choose clusters
n_clusters = st.sidebar.slider("Broj klastera", 2, 20, DEFAULT_CLUSTERS)
labels, centroids = run_clustering(embs, n_clusters)
clients_df["cluster"] = labels


# 7) single-client lookup with emoji table
st.subheader("🔍 Pretraga profila klijenta")
chosen = st.selectbox("Odaberi Klijenta (IDENTIFIKATOR_KLIJENTA):", options=client_ids)

if chosen:
    info = clients_df.loc[clients_df["IDENTIFIKATOR_KLIJENTA"] == chosen].iloc[0]
    prods = prods_df[prods_df["IDENTIFIKATOR_KLIJENTA"] == chosen]
    
    st.markdown(f"### 📇 Klijent: **{chosen}**")
    
    # Build rows: (emoji, label, value)
    rows = []
    # DOB + age
    if pd.notnull(info.get("DOB")):
        dob = pd.to_datetime(info["DOB"]).date()
        age = (pd.Timestamp.today().date() - dob).days // 365
        rows.append(("🎂", "Dob", f"{dob} ({age}g)"))
    # Gender
    if pd.notnull(info.get("SPOL")):
        rows.append(("⚧️", "Spol", info["SPOL"]))
    # City
    if pd.notnull(info.get("GRAD_STALNE_ADRESE")):
        rows.append(("🏙️", "Grad stalne adrese", info["GRAD_STALNE_ADRESE"]))
    # Occupation
    if pd.notnull(info.get("ZANIMANJE")):
        rows.append(("💼", "Zanimanje", info["ZANIMANJE"]))
        # job/position/status
    for fld, emoji, label in [
        ("POZICIJA",   "🏷️", "Pozicija"),
        ("STATUS_ZAPOSLENJA", "📝", "Status zaposlenja")
    ]:
        if pd.notnull(info.get(fld)):
            rows.append((emoji, label, info[fld]))
    # credit rating
    if pd.notnull(info.get("KREDITNI_RATING")):
        rows.append(("⭐", "Kreditni rating", info["KREDITNI_RATING"]))
    # Years of service
    if pd.notnull(info.get("GODINE_STAZA")):
        rows.append(("⏳", "Godine staža", info["GODINE_STAZA"]))
    # Household size
    if pd.notnull(info.get("BROJ_CLANOVA_KUCANSTVA")):
        rows.append(("🏠", "Članova kućanstva", info["BROJ_CLANOVA_KUCANSTVA"]))
    # Credit rating
    if pd.notnull(info.get("KREDITNI_RATING")):
        rows.append(("⭐", "Kreditni rating", info["KREDITNI_RATING"]))
    # Number of products
    rows.append(("📦", "Broj proizvoda", str(len(prods))))

    # Render Markdown table
    md = "|   | Polje                       | Vrijednost |\n"
    md += "|:-:|:----------------------------|:-----------|\n"
    for emoji, label, val in rows:
        md += f"| {emoji} | **{label}** | {val} |\n"
    st.markdown(md)


st.markdown("---")
st.subheader("🎯 Preporuke za klijenta")

recs = model.recommend(chosen, top_n=10)
hexes = ["#004d00","#006600","#008000","#009900","#00b300",
         "#00cc00","#33cc33","#66cc66","#99cc99","#ccffcc"]

html = """
<table style="width:100%; border-collapse: collapse; font-family: sans-serif;">
  <tr>
    <th style="width:10px;"></th>
    <th style="width:40px; text-align:left;">Rank</th>
    <th style="text-align:left;">Proizvod</th>
  </tr>
"""
for i, prod in enumerate(recs):
    color = hexes[i]
    rank  = i+1
    html += f"""
  <tr>
    <td style="padding: 0;">
      <div style="width:10px; height:24px; background:{color};"></div>
    </td>
    <td style="width:40px; padding: 4px 8px;">{rank}</td>
    <td style="padding: 4px 8px;">{prod}</td>
  </tr>
"""
html += "</table>"

st.markdown(html, unsafe_allow_html=True)


st.markdown("---")


# 3) cluster‐size table + bar chart + silhouette
cluster_sizes = (
    clients_df.groupby("cluster")["IDENTIFIKATOR_KLIJENTA"]
    .count()
    .rename("Broj klijenata")
    .reset_index()
)
st.subheader("📊 Klasteri i njihove veličine")
st.dataframe(cluster_sizes, use_container_width=True)


st.markdown("---")

import plotly.graph_objects as go

# 1) Run PCA on embeddings only
pca = PCA(n_components=2)
embs_2d = pca.fit_transform(embs)
df_vis = pd.DataFrame(embs_2d, columns=["PC1", "PC2"])
df_vis["cluster"] = labels.astype(str)
df_vis["client_id"] = client_ids

# 2) Project centroids (drop the age dim)
cent_emb = centroids[:, :embs.shape[1]]
cent2d = pca.transform(cent_emb)
df_cent = pd.DataFrame(cent2d, columns=["PC1", "PC2"])
df_cent["cluster"] = np.arange(n_clusters).astype(str)

# 3) Build a consistent color map (one color per cluster)
clusters = sorted(df_vis["cluster"].unique())
palette = px.colors.qualitative.Plotly  # or choose another Qualitative palette
color_map = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}

# 4) Scatter client points, colored by cluster
fig = px.scatter(
    df_vis,
    x="PC1", y="PC2",
    color="cluster",
    color_discrete_map=color_map,
    hover_data=["client_id"],
    title="PCA projekcija klijenata po klasterima",
    template="plotly_white",
)

# 5) Overlay centroids as big X's in matching colors
for cluster_id, row in df_cent.groupby("cluster"):
    fig.add_trace(
        go.Scatter(
            x=[row.PC1.values[0]],
            y=[row.PC2.values[0]],
            mode="markers",
            marker=dict(
                symbol="x",
                size=18,
                color=color_map[cluster_id],
                line=dict(color="black", width=2),
            ),
            name=f"Centroid {cluster_id}",
            showlegend=True,
        )
    )

# 6) Tidy up
fig.update_layout(
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    legend_title="Klaster",
    margin=dict(t=50, b=20, l=20, r=20)
)

# 6a) Reorder so Centroids are drawn on top
centroid_traces = [t for t in fig.data if t.name.startswith("Centroid")]
other_traces    = [t for t in fig.data if not t.name.startswith("Centroid")]
fig.data = other_traces + centroid_traces

# 6b) (Optional) bump centroid size & full opacity for extra clarity
for t in centroid_traces:
    t.marker.size = 20
    t.marker.opacity = 1.0

# 7) Render
st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

# 5) popular products per cluster
st.subheader("🏷️ Popularni proizvodi po klasteru")

if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = 0
    
selected_cluster = st.selectbox(
    "Odaberi klaster za detalje",
    options=sorted(cluster_sizes["cluster"].tolist()),
     key="selected_cluster",
)
cluster_clients = clients_df.loc[
    clients_df["cluster"] == selected_cluster, "IDENTIFIKATOR_KLIJENTA"
].tolist()
all_recs = []
for cid in cluster_clients:
    all_recs.extend(model.recommend(cid, top_n=5))
top5 = Counter(all_recs).most_common(5)
df_top5 = pd.DataFrame(top5, columns=["Proizvod","Učestalost"])
st.table(df_top5)


#-------------------------------------
# … after you compute `cluster_clients` …
cluster_df = clients_df[clients_df["cluster"] == selected_cluster].copy()


# 1) DOB → AGE

#st.write(type(cluster_df["DOB"][0]))
cluster_df["age"] = cluster_df["DOB"].astype(int)

# … after you’ve created cluster_df["age"] …

st.subheader(f"🔍 Statistika za klaster {selected_cluster}")
st.markdown("**Dobna distribucija**")
# basic age stats
st.write(f"- Prosječna dob: {cluster_df['age'].mean():.1f} godina")
st.write(f"- Medijan: {cluster_df['age'].median():.1f} godina")


import plotly.graph_objects as go
from scipy.stats import gaussian_kde



# 2) GENDER

    
import plotly.express as px


fig_age = px.histogram(
    cluster_df,
    x="age",
    color="SPOL",           # use your actual column name
    nbins=20,
    labels={"age": "Dob (godine)", "SPOL": "Spol"},
    title="Distribucija spola po dobi",
    color_discrete_map={     # map each category to the requested colour
        "F": "lightpink",
        "M": "lightblue",
        "O": "yellow"
    },
)
fig_age.update_traces(
    marker_line_width=1,     # thin white border to slim bars
    marker_line_color="white"
)
fig_age.update_layout(
    bargap=0.1               # small gap between bars
)
st.plotly_chart(fig_age, use_container_width=True)


if "SPOL" in cluster_df:
    fig_sex = px.pie(
        cluster_df, names="SPOL",
        title="Udjeli muškaraca/žena"
    )
    st.plotly_chart(fig_sex, use_container_width=True)
    
    
# 3) JOB / POSITION
job_col = None
for c in ("ZANIMANJE", "POZICIJA", "STATUS_ZAPOSLENJA"):
    if c in cluster_df:
        job_col = c
        break
if job_col:
    st.markdown(f"**Distribucija `{job_col}`**")
    job_counts = (
        cluster_df[job_col]
        .value_counts()
        .rename_axis(job_col)
        .reset_index(name="count")
    )
    fig_job = px.bar(
        job_counts, x=job_col, y="count",
        title=f"Broj klijenata po zanimanju"
    )
    st.plotly_chart(fig_job, use_container_width=True)

# 4) PRODUCTS PER CLIENT
st.markdown("**Broj proizvoda po klijentu**")
prod_counts = (
    prods_df[prods_df["IDENTIFIKATOR_KLIJENTA"].isin(cluster_clients)]
    .groupby("IDENTIFIKATOR_KLIJENTA")
    .size()
    .rename("num_products")
    .reset_index()
)
avg_prod = prod_counts["num_products"].mean()
st.write(f"- Prosječno: {avg_prod:.2f} proizvoda po klijentu")

import plotly.express as px



fig_ppc = px.histogram(
    prod_counts,
    x="num_products",
    nbins=10,
    labels={"num_products": "Broj proizvoda"},
    title="Raspodjela broja proizvoda",
    color_discrete_sequence=["#a0d88f"],    # plava nijansa
)
# add a thin white border to each bar to make them appear slimmer
fig_ppc.update_traces(
    marker_line_width=1,
    marker_line_color="white",
)
# shrink the padding between bars (0.0–1.0 scale; default is 0.2)
fig_ppc.update_layout(
    bargap=0.1,
)
st.plotly_chart(fig_ppc, use_container_width=True)


# 5) CREDIT RATING
if "KREDITNI_RATING" in cluster_df:
    fig_cr = px.histogram(
        cluster_df,
        x="KREDITNI_RATING",
        title="Distribucija kreditnog ratinga",
        color_discrete_sequence=["#ff7f0e"],    # narančasta nijansa
    )
    fig_cr.update_traces(
        marker_line_width=1,
        marker_line_color="white",
    )
    fig_cr.update_layout(
        bargap=0.1,
    )
    st.plotly_chart(fig_cr, use_container_width=True)

