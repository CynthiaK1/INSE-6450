"""
Feature engineering: compute robustness features from adversarial telemetry.
Produces per-(attack_type, epsilon) and per-class aggregated features.
"""

import os
import yaml
import numpy as np
import pandas as pd


def compute_robustness_features(telemetry_df):
    """
    Compute aggregated robustness features from raw telemetry.

    Returns:
        features_df: One row per (attack_type, epsilon) with robustness metrics.
        class_features_df: Per-class breakdown.
    """
    features = []

    # Group by attack configuration
    for (attack_type, epsilon), group in telemetry_df.groupby(["attack_type", "epsilon"]):
        n = len(group)

        # Accuracy-based features
        clean_correct = (group["true_label"] == group["clean_pred"]).sum()
        adv_correct = (group["true_label"] == group["adv_pred"]).sum()
        clean_acc = clean_correct / n
        adv_acc = adv_correct / n
        acc_drop = clean_acc - adv_acc

        # Confidence-based features
        mean_clean_conf = group["clean_conf"].mean()
        mean_adv_conf = group["adv_conf"].mean()
        conf_drop = mean_clean_conf - mean_adv_conf
        adv_conf_std = group["adv_conf"].std()

        # Attack success features
        asr = group["flipped"].mean()

        # Per-class ASR
        class_asr = group.groupby("true_label")["flipped"].mean()
        worst_class_asr = class_asr.max()
        best_class_asr = class_asr.min()
        asr_std = class_asr.std()

        # Perturbation features
        mean_l_inf = group["l_inf_norm"].mean()

        features.append({
            "attack_type": attack_type,
            "epsilon": epsilon,
            "clean_accuracy": round(clean_acc, 6),
            "adversarial_accuracy": round(adv_acc, 6),
            "accuracy_drop": round(acc_drop, 6),
            "mean_clean_confidence": round(mean_clean_conf, 6),
            "mean_adv_confidence": round(mean_adv_conf, 6),
            "confidence_drop": round(conf_drop, 6),
            "adv_confidence_std": round(adv_conf_std, 6),
            "attack_success_rate": round(asr, 6),
            "worst_class_asr": round(worst_class_asr, 6),
            "best_class_asr": round(best_class_asr, 6),
            "class_asr_std": round(asr_std, 6),
            "mean_l_inf_norm": round(mean_l_inf, 6),
            "num_samples": n,
        })

    features_df = pd.DataFrame(features)

    # Compute per-class features
    class_features = []
    for (attack_type, epsilon, label), group in telemetry_df.groupby(
        ["attack_type", "epsilon", "true_label"]
    ):
        n = len(group)
        class_features.append({
            "attack_type": attack_type,
            "epsilon": epsilon,
            "true_label": label,
            "class_clean_acc": (group["true_label"] == group["clean_pred"]).mean(),
            "class_adv_acc": (group["true_label"] == group["adv_pred"]).mean(),
            "class_asr": group["flipped"].mean(),
            "class_mean_adv_conf": group["adv_conf"].mean(),
            "num_samples": n,
        })

    class_features_df = pd.DataFrame(class_features)

    return features_df, class_features_df


def compute_decay_rate(features_df):
    """
    Compute robustness decay rate: slope of accuracy vs epsilon.
    A steeper negative slope = more vulnerable model.
    """
    decay_rates = []
    for attack_type, group in features_df.groupby("attack_type"):
        group = group.sort_values("epsilon")
        epsilons = group["epsilon"].values
        accs = group["adversarial_accuracy"].values

        if len(epsilons) >= 2:
            # Linear regression slope
            slope = np.polyfit(epsilons, accs, 1)[0]
        else:
            slope = 0.0

        decay_rates.append({
            "attack_type": attack_type,
            "decay_rate": round(slope, 6),
        })

    return pd.DataFrame(decay_rates)


def main():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load telemetry
    telemetry_df = pd.read_csv(config["outputs"]["telemetry_path"])
    print(f"Loaded telemetry: {len(telemetry_df)} records")

    # Compute features
    features_df, class_features_df = compute_robustness_features(telemetry_df)
    decay_df = compute_decay_rate(features_df)

    # Save
    os.makedirs(os.path.dirname(config["outputs"]["features_path"]), exist_ok=True)
    features_df.to_csv(config["outputs"]["features_path"], index=False)
    class_features_df.to_csv(
        config["outputs"]["features_path"].replace(".csv", "_per_class.csv"),
        index=False
    )
    decay_df.to_csv(
        config["outputs"]["features_path"].replace(".csv", "_decay.csv"),
        index=False
    )

    print(f"\nRobustness Features:\n{features_df.to_string(index=False)}")
    print(f"\nDecay Rates:\n{decay_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
