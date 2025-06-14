# Helper functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, classification_report
from sklearn.metrics import ndcg_score

def infer_num_cat(df, threshold=0.9):
    num_cols = []
    cat_cols = [] 
    for col in df.columns:
        series = df[col]
        if is_numeric_dtype(series):
            num_cols.append(col)
            continue
        
        coerced = pd.to_numeric(series, errors='coerce')
        frac_numeric = coerced.notna().mean()
        if frac_numeric >= threshold:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    
    return num_cols, cat_cols


def fill_missing_values(df, num_cols, cat_cols):
    # Filling missing values
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna('NA')

    return df

def scale_numeric_values(df, num_cols):
    df_num = df[num_cols].copy()
    df_num = df_num.apply(pd.to_numeric, errors='coerce')
    df_num = df_num.fillna(0)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_num)

    return X_num

def scale_categorical_values(df, cat_cols):

    le_dict = {}
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=int)
    cat_dims = []
    
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        codes = le.fit_transform(df[col].astype(str))
        X_cat[:, i] = codes
        cat_dims.append(len(le.classes_))
        le_dict[col] = le

    return X_cat, cat_dims


def compute_classification_metrics(y_true, y_pred, average='binary', zero_division=0):

    precision = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=zero_division)
    jaccard = jaccard_score(y_true, y_pred, average=average, zero_division=zero_division)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard
    }

def print_classification_report(y_true, y_pred, target_names=None, zero_division=0):

    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=zero_division
    ))



def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:

    idx = np.argsort(-y_score)[:k]
    return y_true[idx].sum() / k

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:

    total_relevant = y_true.sum()
    if total_relevant == 0:
        return 0.0
    idx = np.argsort(-y_score)[:k]
    return y_true[idx].sum() / total_relevant

def average_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    idx = np.argsort(-y_score)[:k]
    hits = y_true[idx]
    if hits.sum() == 0:
        return 0.0
    precisions = [ hits[:i+1].sum()/(i+1) for i in range(k) if hits[i] ]
    return np.mean(precisions)

def mean_average_precision(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    return np.mean([
        average_precision_at_k(y_true[i], y_score[i], k)
        for i in range(len(y_true))
    ])

def hit_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    idx = np.argsort(-y_score, axis=1)[:, :k]
    hits = np.array([ y_true[i, idx[i]].sum() > 0 for i in range(y_true.shape[0]) ])
    return hits.mean()

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true2 = y_true.reshape(1, -1) if y_true.ndim == 1 else y_true
    y_score2 = y_score.reshape(1, -1) if y_score.ndim == 1 else y_score
    return ndcg_score(y_true2, y_score2, k=k)