# fintech_hpb/src/models/final_embedding.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.helper_functions import (
    infer_num_cat,
    fill_missing_values,
    scale_numeric_values,
    scale_categorical_values,
    compute_classification_metrics,
    print_classification_report,
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    mean_average_precision,
    hit_rate_at_k,
    ndcg_at_k,
)


class EmbeddingRecommender(nn.Module):
    def __init__(
        self,
        client_df: pd.DataFrame,
        produkti_df: pd.DataFrame,
        embedding_dim: int = 32,
        alpha_pop: float = 0.5,
        cutoff_date: str = "2023-12-22",
        device: str = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.alpha_pop = alpha_pop

        self.client_ids = client_df["IDENTIFIKATOR_KLIJENTA"].unique().tolist()
        self.prod_ids   = produkti_df["NAZIV_VRSTE_PROIZVODA"].unique().tolist()
        self.C, self.P  = len(self.client_ids), len(self.prod_ids)
        self.client2idx = {cid: i for i, cid in enumerate(self.client_ids)}
        self.prod2idx   = {pid: i for i, pid in enumerate(self.prod_ids)}

        produkti_df["datum_otvaranja_parsed"] = pd.to_datetime(
            produkti_df["datum_otvaranja_parsed"],
            dayfirst=True, errors="coerce"
        )
        cutoff = pd.to_datetime(cutoff_date)
        past_mask = produkti_df["datum_otvaranja_parsed"] <= cutoff
        pop_counts = (
            produkti_df.loc[past_mask, "NAZIV_VRSTE_PROIZVODA"]
            .value_counts()
            .reindex(self.prod_ids, fill_value=0)
        )
        pop = pop_counts.values.astype(float)
        pop = pop / pop.sum()
        self.pop_scores = torch.tensor(pop, dtype=torch.float, device=self.device)

        num_u, cat_u = infer_num_cat(client_df)
        cdf = fill_missing_values(client_df.copy(), num_u, cat_u)
        Xu_num = scale_numeric_values(cdf, num_u)
        Xu_cat, dim_u = scale_categorical_values(cdf, cat_u)

        self.num_proj = nn.Linear(Xu_num.shape[1], embedding_dim)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(d, embedding_dim) for d in dim_u
        ])
        self.cat_proj = nn.Linear(len(dim_u)*embedding_dim, embedding_dim)

        self.Xu_num = torch.tensor(Xu_num, dtype=torch.float, device=self.device)
        self.Xu_cat = torch.tensor(Xu_cat, dtype=torch.long, device=self.device)

        hierarchy_cols = [
            "NAZIV_DOMENE_PROIZVODA",
            "NAZIV_KATEGORIJE_PROIZVODA",
            "NAZIV_KLASE_PROIZVODA",
            "NAZIV_GRUPE_PROIZVODA",
            "NAZIV_VRSTE_PROIZVODA",
        ]
        prod_hier = (
            produkti_df
            .groupby("NAZIV_VRSTE_PROIZVODA", sort=False)[hierarchy_cols]
            .first()
            .reindex(self.prod_ids)
        )
        prod_hier = prod_hier.reset_index(drop=True)

        num_p, cat_p = infer_num_cat(prod_hier)
        pdf = fill_missing_values(prod_hier.copy(), num_p, cat_p)
        Xp_num = scale_numeric_values(pdf, num_p) if num_p else None
        Xp_cat, dim_p = scale_categorical_values(pdf, cat_p)

        if Xp_num is not None:
            self.prod_num_proj = nn.Linear(Xp_num.shape[1], embedding_dim)
            self.Xp_num = torch.tensor(Xp_num, dtype=torch.float, device=self.device)
        else:
            self.prod_num_proj = None

        self.prod_cat_embeds = nn.ModuleList([
            nn.Embedding(d, embedding_dim) for d in dim_p
        ])
        self.prod_cat_proj = nn.Linear(len(dim_p)*embedding_dim, embedding_dim)
        self.Xp_cat = torch.tensor(Xp_cat, dtype=torch.long, device=self.device)

        self.to(self.device)

    def encode_clients(self) -> torch.Tensor:
        num_e = self.num_proj(self.Xu_num)
        cat_es = [emb(self.Xu_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_e  = torch.cat(cat_es, dim=1)
        cat_e  = self.cat_proj(cat_e)
        return torch.relu(num_e + cat_e)

    def encode_products(self) -> torch.Tensor:
        if self.prod_num_proj:
            num_e = self.prod_num_proj(self.Xp_num)
        else:
            num_e = torch.zeros((self.P, self.embedding_dim), device=self.device)
        cat_es = [emb(self.Xp_cat[:, i]) for i, emb in enumerate(self.prod_cat_embeds)]
        cat_e  = torch.cat(cat_es, dim=1)
        cat_e  = self.prod_cat_proj(cat_e)
        return torch.relu(num_e + cat_e)

    def forward(self, c_idx, pos_idx, neg_idx) -> torch.Tensor:
        u_e = self.encode_clients()[c_idx]
        V   = self.encode_products()
        p_e = V[pos_idx]
        n_e = V[neg_idx]
        pos = (u_e * p_e).sum(dim=1)
        neg = (u_e * n_e).sum(dim=1)
        return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

    def train_bpr(self, interactions: pd.DataFrame,
                  epochs: int = 10, batch_size: int = 512, lr: float = 1e-3):
        pairs = [
            (self.client2idx[c], self.prod2idx[p])
            for c, p in zip(
                interactions["IDENTIFIKATOR_KLIJENTA"],
                interactions["NAZIV_VRSTE_PROIZVODA"],
            )
        ]
        opt = optim.Adam(self.parameters(), lr=lr)
        for ep in range(1, epochs+1):
            np.random.shuffle(pairs)
            total = 0.0
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                if not batch:
                    break
                c_idx, p_pos = zip(*batch)
                p_neg = np.random.randint(0, self.P, size=len(batch))

                c_t = torch.tensor(c_idx, dtype=torch.long, device=self.device)
                pp  = torch.tensor(p_pos, dtype=torch.long, device=self.device)
                pn  = torch.tensor(p_neg, dtype=torch.long, device=self.device)

                opt.zero_grad()
                loss = self(c_t, pp, pn)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)

            print(f"Epoch {ep}/{epochs} — avg loss {total/len(pairs):.4f}")

    def recommend(self, client_id, top_n: int = 10):
        u_i = self.client2idx[client_id]
        u   = self.encode_clients()[u_i]
        V   = self.encode_products()
        emb = (V @ u).detach().cpu().numpy()
        pop = self.pop_scores.cpu().numpy()
        scores = self.alpha_pop * pop + (1-self.alpha_pop) * emb
        idx = np.argsort(-scores)[:top_n]
        return [self.prod_ids[j] for j in idx]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    data_dir = root / "hpb_data"
    print("Data directory:", data_dir)

    prod_df = pd.read_csv(data_dir / "PROIZVODI_CLEANED.csv", low_memory=False)
    prod_df["datum_otvaranja_parsed"] = pd.to_datetime(
        prod_df["datum_otvaranja_parsed"], errors="coerce"
    )
    cutoff = pd.Timestamp("2023-12-22")
    past   = prod_df[prod_df["datum_otvaranja_parsed"] <= cutoff].copy()
    future = prod_df[prod_df["datum_otvaranja_parsed"] >  cutoff].copy()

    clients_df = pd.read_csv(data_dir / "KLIJENTI_CLEANED.csv")

    model = EmbeddingRecommender(
        client_df=clients_df,
        produkti_df=prod_df,
        embedding_dim=32,
        alpha_pop=0.5,
        cutoff_date="2023-12-22",
    )
    model.train_bpr(interactions=past, epochs=10, batch_size=512, lr=3e-4)


    cids   = clients_df["IDENTIFIKATOR_KLIJENTA"].tolist()
    prods  = model.prod_ids
    N, P   = len(cids), len(prods)
    TOP_K  = 10

    cp_fut      = pd.crosstab(
        future["IDENTIFIKATOR_KLIJENTA"],
        future["NAZIV_VRSTE_PROIZVODA"],
    ).reindex(index=cids, columns=prods, fill_value=0)

    y_true      = np.zeros((N, P), dtype=int)
    scores_mat  = np.zeros((N, P), dtype=float)


    client_embs = model.encode_clients().detach().cpu().numpy()
    prod_embs   = model.encode_products().detach().cpu().numpy()
    pop         = model.pop_scores.cpu().numpy()

    for i, cid in enumerate(cids):
        relevant = cp_fut.columns[cp_fut.loc[cid] > 0].tolist()
        for j, p in enumerate(prods):
            y_true[i, j] = int(p in relevant)

        emb_scores       = client_embs[i] @ prod_embs.T  
        scores_mat[i, :] = model.alpha_pop * pop + (1 - model.alpha_pop) * emb_scores

    y_pred = np.zeros_like(y_true)
    for i in range(N):
        top_idx      = np.argsort(-scores_mat[i])[:TOP_K]
        y_pred[i, top_idx] = 1

    cls_metrics = compute_classification_metrics(
        y_true, y_pred, average="samples", zero_division=0
    )
    print("\n=== Precision/Recall/F1/Jaccard @10 ===")
    for name, val in cls_metrics.items():
        print(f"{name:>8}: {val:.4f}")

    print("\n=== Detailed per-product report ===")
    print_classification_report(
        y_true, y_pred, target_names=prods, zero_division=0
    )

    map10  = mean_average_precision(y_true, scores_mat, k=TOP_K)
    hr10   = hit_rate_at_k(y_true, scores_mat, k=TOP_K)
    ndcg10 = ndcg_at_k(y_true, scores_mat, k=TOP_K)

    print(f"\nMAP@{TOP_K}:  {map10:.4f}")
    print(f"HitRate@{TOP_K}: {hr10:.4f}")
    print(f"NDCG@{TOP_K}:    {ndcg10:.4f}")