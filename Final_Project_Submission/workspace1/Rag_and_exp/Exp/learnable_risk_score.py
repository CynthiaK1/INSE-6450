"""
learnable_risk_score.py — H4 extension.

Replaces hand-picked risk-score weights with learnable ones, and adds a
smarter gate (logistic meta-classifier) over the diagnose features.

Two models:
  1. LearnableRiskScore: convex combination of (Δacc, Δconf, ASR, WCASR),
     trained to predict deployment-failure label via BCE.
     Constraint: weights are softmaxed so they sum to 1 and are interpretable.

  2. SmarterGate: logistic regression meta-classifier over the same four
     features plus the three P(X,Y) divergence components. Outputs calibrated
     P(failure) used directly as the gate signal. Threshold τ is then
     calibrated for a target false-block rate (e.g., FPR ≤ 5%).

Train on a labeled set of (window_features, did_it_fail_in_production?)
tuples. For your paper you can construct this set by treating drift severity
> threshold as the failure label, OR by using the actual post-mitigation
performance drop > X% as the label.

Usage:
    python learnable_risk_score.py --csv risk_features.csv --target failed
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split


class LearnableRiskScore(nn.Module):
    """Convex combination of risk components with softmax-constrained weights."""
    def __init__(self, n_components=4):
        super().__init__()
        self.raw_w = nn.Parameter(torch.zeros(n_components))

    @property
    def weights(self):
        return torch.softmax(self.raw_w, dim=0)

    def forward(self, x):
        # x: (B, n_components) — already in [0, 1] (you should normalize before)
        return (x * self.weights).sum(dim=1)


def train_learnable_score(X, y, epochs=500, lr=0.05):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = LearnableRiskScore(n_components=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        score = model(X_t)
        # Use logistic loss on the score (treat score as a logit-like signal)
        loss = nn.functional.binary_cross_entropy(
            torch.clamp(score, 1e-6, 1 - 1e-6), y_t
        )
        loss.backward()
        opt.step()
    return model


def smarter_gate(X_train, y_train, X_test, y_test, target_fpr=0.05):
    """
    Logistic regression over diagnose features. Returns the calibrated
    threshold that achieves the target false-positive (false-block) rate.
    """
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    p_train = clf.predict_proba(X_train)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    # Calibrate threshold on the negative class (passes that should remain passes)
    neg_scores = p_train[y_train == 0]
    if len(neg_scores) == 0:
        tau = 0.5
    else:
        tau = float(np.quantile(neg_scores, 1 - target_fpr))

    return {
        "model": clf,
        "tau": tau,
        "auc": roc_auc_score(y_test, p_test),
        "brier": brier_score_loss(y_test, p_test),
        "test_block_rate": float((p_test >= tau).mean()),
        "test_actual_fpr": float(((p_test >= tau) & (y_test == 0)).sum()
                                  / max((y_test == 0).sum(), 1)),
        "test_actual_tpr": float(((p_test >= tau) & (y_test == 1)).sum()
                                  / max((y_test == 1).sum(), 1)),
        "coefficients": dict(zip(
            [f"feat_{i}" for i in range(X_train.shape[1])],
            clf.coef_[0].tolist()
        )),
    }


def main(csv, target_col, score_cols, gate_cols):
    df = pd.read_csv(csv)
    y = df[target_col].astype(int).values
    X_score = df[score_cols].values
    X_gate = df[gate_cols].values

    X_score = (X_score - X_score.min(0)) / (X_score.ptp(0) + 1e-9)

    print("=== Learnable risk score (H4 extension) ===")
    model = train_learnable_score(X_score, y)
    learned_w = model.weights.detach().numpy()
    print("Learned weights (sum=1):")
    for name, w in zip(score_cols, learned_w):
        print(f"  {name}: {w:.4f}")
    auc = roc_auc_score(y, model(torch.tensor(X_score, dtype=torch.float32)).detach().numpy())
    print(f"Train AUC: {auc:.4f}")

    print("\n=== Smarter gate (logistic meta-classifier over diagnose features) ===")
    Xtr, Xte, ytr, yte = train_test_split(X_gate, y, test_size=0.3,
                                          random_state=42, stratify=y)
    res = smarter_gate(Xtr, ytr, Xte, yte, target_fpr=0.05)
    print(f"Test AUC: {res['auc']:.4f}")
    print(f"Brier score: {res['brier']:.4f}")
    print(f"Calibrated tau (target FPR=5%): {res['tau']:.4f}")
    print(f"Test actual FPR: {res['test_actual_fpr']:.4f}")
    print(f"Test actual TPR: {res['test_actual_tpr']:.4f}")
    print(f"Test block rate: {res['test_block_rate']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", default="failed")
    p.add_argument("--score_cols", nargs="+",
                   default=["delta_acc", "delta_conf", "asr", "wcasr"])
    p.add_argument("--gate_cols", nargs="+",
                   default=["delta_acc", "delta_conf", "asr", "wcasr",
                            "d_px", "d_pyx", "d_py"])
    args = p.parse_args()
    main(args.csv, args.target, args.score_cols, args.gate_cols)
