"""
mil_baselines.py — additional baselines for the attention-MIL attribution experiment (H3).

Adds three baselines beyond per-feature PSI:
  1. Isolation Forest anomaly score
  2. Classifier Two-Sample Test (Lipton-style): train logistic regression to
     distinguish buffer from window, rank window samples by predicted prob.
  3. KNN distance to buffer (mean distance to k nearest buffer points)

Then runs N trials of injected ground-truth drift and compares all methods
to attention-MIL via paired Wilcoxon signed-rank test.

Usage:
    python mil_baselines.py --n_trials 20 --n_window 500 --n_drift 50 --noise 2.5
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_adult():
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    X = data.data.select_dtypes(include=[np.number]).fillna(0).values
    return StandardScaler().fit_transform(X)


def make_trial(X_pool, n_buffer, n_window, n_drift, noise, rng):
    idx = rng.permutation(len(X_pool))
    buffer = X_pool[idx[:n_buffer]]
    window_clean = X_pool[idx[n_buffer:n_buffer + n_window - n_drift]]
    drift_base = X_pool[idx[n_buffer + n_window - n_drift:n_buffer + n_window]]
    drift = drift_base + rng.normal(0, noise, size=drift_base.shape)
    window = np.vstack([window_clean, drift])
    truth = np.concatenate([np.zeros(len(window_clean)), np.ones(len(drift))])
    perm = rng.permutation(len(window))
    return buffer, window[perm], truth[perm]


def recall_at_k(scores, truth, k):
    top_k = np.argsort(-scores)[:k]
    return truth[top_k].sum() / truth.sum()


def psi_baseline(buffer, window, n_bins=10):
    scores = np.zeros(len(window))
    for f in range(buffer.shape[1]):
        edges = np.quantile(buffer[:, f], np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        b_hist, _ = np.histogram(buffer[:, f], bins=edges)
        w_hist, _ = np.histogram(window[:, f], bins=edges)
        b_p = (b_hist + 1) / (b_hist.sum() + n_bins)
        w_p = (w_hist + 1) / (w_hist.sum() + n_bins)
        psi_per_bin = (w_p - b_p) * np.log(w_p / b_p)
        bin_idx = np.clip(np.digitize(window[:, f], edges) - 1, 0, n_bins - 1)
        scores += np.abs(psi_per_bin[bin_idx])
    return scores


def isoforest_baseline(buffer, window, rng):
    iso = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=rng.integers(1e6))
    iso.fit(buffer)
    return -iso.score_samples(window)


def c2st_baseline(buffer, window, rng):
    X = np.vstack([buffer, window])
    y = np.concatenate([np.zeros(len(buffer)), np.ones(len(window))])
    clf = LogisticRegression(max_iter=1000, random_state=rng.integers(1e6))
    clf.fit(X, y)
    return clf.predict_proba(window)[:, 1]


def knn_baseline(buffer, window, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(buffer)
    dist, _ = nn.kneighbors(window)
    return dist.mean(axis=1)


def attention_mil_proxy(buffer, window):
    """
    Differentiable attention proxy: weight each window sample by its
    Mahalanobis-like distance to the buffer mean, softmaxed. This is a
    drop-in stand-in for your trained attention-MIL aggregator so the
    comparison runs end-to-end without requiring the trained module here.
    Replace with the actual MIL forward pass when integrating into your repo.
    """
    mu = buffer.mean(axis=0)
    cov_inv = np.linalg.pinv(np.cov(buffer.T) + 1e-3 * np.eye(buffer.shape[1]))
    diff = window - mu
    md = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    return md


def run(n_trials, n_window, n_drift, noise, n_buffer, k, seed):
    X = load_adult()
    rng = np.random.default_rng(seed)
    methods = ["psi", "isoforest", "c2st", "knn", "attention_mil"]
    results = {m: [] for m in methods}

    for _ in range(n_trials):
        buf, win, truth = make_trial(X, n_buffer, n_window, n_drift, noise, rng)
        results["psi"].append(recall_at_k(psi_baseline(buf, win), truth, k))
        results["isoforest"].append(recall_at_k(isoforest_baseline(buf, win, rng), truth, k))
        results["c2st"].append(recall_at_k(c2st_baseline(buf, win, rng), truth, k))
        results["knn"].append(recall_at_k(knn_baseline(buf, win), truth, k))
        results["attention_mil"].append(recall_at_k(attention_mil_proxy(buf, win), truth, k))

    df = pd.DataFrame(results)
    summary = df.agg(["mean", "std"]).T
    print("\n=== Recall@{} over {} trials ===".format(k, n_trials))
    print(summary.round(4))

    print("\n=== Paired Wilcoxon: attention_mil vs each baseline ===")
    rows = []
    for m in ["psi", "isoforest", "c2st", "knn"]:
        diffs = df["attention_mil"] - df[m]
        if (diffs == 0).all():
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(df["attention_mil"], df[m], zero_method="wilcox")
        effect = diffs.mean() / (diffs.std() + 1e-12)
        rows.append({"baseline": m, "mean_diff": diffs.mean(),
                     "wilcoxon_stat": stat, "p_value": p, "cohen_d": effect})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    df.to_csv("mil_results.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_window", type=int, default=500)
    p.add_argument("--n_drift", type=int, default=50)
    p.add_argument("--noise", type=float, default=2.5)
    p.add_argument("--n_buffer", type=int, default=2000)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(args.n_trials, args.n_window, args.n_drift, args.noise,
        args.n_buffer, args.k, args.seed)
