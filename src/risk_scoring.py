"""
Compute the Robustness Risk Score and enforce deploy/block gate.
Aggregates features into a single composite risk metric.
"""

import os
import yaml
import pandas as pd


def compute_risk_score(features_df, weights):
    """
    Compute composite Robustness Risk Score per (attack_type, epsilon).

    Risk Score = weighted combination of:
        - accuracy_drop (normalized to [0,1])
        - confidence_drop (normalized to [0,1])
        - attack_success_rate (already in [0,1])
        - worst_class_asr (already in [0,1])

    Higher score = higher risk = less robust.
    """
    risk_records = []

    for _, row in features_df.iterrows():
        # Normalize accuracy drop (max possible = 1.0)
        norm_acc_drop = min(row["accuracy_drop"], 1.0)
        # Normalize confidence drop (max possible = 1.0)
        norm_conf_drop = min(max(row["confidence_drop"], 0.0), 1.0)
        asr = row["attack_success_rate"]
        worst_asr = row["worst_class_asr"]

        risk_score = (
            weights["accuracy_drop"] * norm_acc_drop
            + weights["confidence_drop"] * norm_conf_drop
            + weights["attack_success_rate"] * asr
            + weights["worst_class_asr"] * worst_asr
        )

        risk_records.append({
            "attack_type": row["attack_type"],
            "epsilon": row["epsilon"],
            "accuracy_drop": row["accuracy_drop"],
            "confidence_drop": row["confidence_drop"],
            "attack_success_rate": asr,
            "worst_class_asr": worst_asr,
            "risk_score": round(risk_score, 6),
        })

    return pd.DataFrame(risk_records)


def enforce_gate(risk_df, threshold):
    """
    Apply deploy/block gate decision based on risk threshold.
    Returns the risk dataframe with a 'gate_decision' column.
    """
    risk_df = risk_df.copy()
    risk_df["gate_decision"] = risk_df["risk_score"].apply(
        lambda s: "BLOCK" if s > threshold else "PASS"
    )
    return risk_df


def generate_risk_report(risk_df, config):
    """Generate and print the final risk report."""
    threshold = config["risk_scoring"]["deploy_threshold"]

    print("=" * 70)
    print("ROBUSTOPS RISK REPORT")
    print("=" * 70)
    print(f"Deploy Threshold: {threshold}")
    print(f"Model: {config['model']['architecture']} on {config['data']['dataset']}")
    print("-" * 70)

    for _, row in risk_df.iterrows():
        gate = row["gate_decision"]
        marker = "✓" if gate == "PASS" else "✗"
        print(
            f"  [{marker}] {row['attack_type'].upper()} ε={row['epsilon']:.3f} | "
            f"Risk: {row['risk_score']:.4f} | "
            f"ASR: {row['attack_success_rate']:.3f} | "
            f"AccDrop: {row['accuracy_drop']:.3f} | "
            f"Gate: {gate}"
        )

    print("-" * 70)
    max_risk = risk_df["risk_score"].max()
    overall = "BLOCK" if max_risk > threshold else "PASS"
    print(f"  Overall Max Risk: {max_risk:.4f} → {overall}")
    print("=" * 70)

    return risk_df


def main():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load features
    features_df = pd.read_csv(config["outputs"]["features_path"])

    # Compute risk scores
    weights = config["risk_scoring"]["weights"]
    risk_df = compute_risk_score(features_df, weights)

    # Apply gate
    threshold = config["risk_scoring"]["deploy_threshold"]
    risk_df = enforce_gate(risk_df, threshold)

    # Generate report
    risk_df = generate_risk_report(risk_df, config)

    # Save
    os.makedirs(os.path.dirname(config["outputs"]["risk_report_path"]), exist_ok=True)
    risk_df.to_csv(config["outputs"]["risk_report_path"], index=False)
    print(f"\nRisk report saved to {config['outputs']['risk_report_path']}")


if __name__ == "__main__":
    main()
