"""
Exploratory Data Analysis: generate all plots required for Milestone 1.
Covers CIFAR-10 data profiling and adversarial telemetry analysis.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms


def plot_cifar10_class_distribution(data_dir, plots_dir):
    """Plot class distribution of CIFAR-10 test set."""
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
    labels = [test_set[i][1] for i in range(len(test_set))]
    class_names = test_set.classes

    fig, ax = plt.subplots(figsize=(10, 5))
    counts = pd.Series(labels).value_counts().sort_index()
    ax.bar(range(10), counts.values, color="steelblue", edgecolor="black")
    ax.set_xticks(range(10))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("CIFAR-10 Test Set: Class Distribution")
    ax.set_ylim(0, 1200)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 15, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cifar10_class_distribution.png"), dpi=150)
    plt.close()
    print("Saved: cifar10_class_distribution.png")


def plot_pixel_distribution(data_dir, plots_dir):
    """Plot pixel intensity distribution across RGB channels."""
    transform = transforms.ToTensor()
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Sample 1000 images for efficiency
    pixels = {"R": [], "G": [], "B": []}
    for i in range(1000):
        img, _ = test_set[i]
        pixels["R"].extend(img[0].flatten().numpy())
        pixels["G"].extend(img[1].flatten().numpy())
        pixels["B"].extend(img[2].flatten().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["red", "green", "blue"]
    for ax, (channel, vals), color in zip(axes, pixels.items(), colors):
        ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor="black")
        ax.set_title(f"{channel} Channel")
        ax.set_xlabel("Pixel Value (normalized)")
        ax.set_ylabel("Frequency")
    plt.suptitle("CIFAR-10 Test Set: Pixel Intensity Distributions (n=1000)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pixel_distribution.png"), dpi=150)
    plt.close()
    print("Saved: pixel_distribution.png")


def plot_sample_images(data_dir, plots_dir):
    """Plot sample images from each class."""
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
    class_names = test_set.classes

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(test_set)):
        _, label = test_set[idx]
        if len(class_indices[label]) < 1:
            class_indices[label].append(idx)
        if all(len(v) >= 1 for v in class_indices.values()):
            break

    for i, ax in enumerate(axes.flat):
        idx = class_indices[i][0]
        img, label = test_set[idx]
        ax.imshow(img)
        ax.set_title(class_names[label], fontsize=10)
        ax.axis("off")

    plt.suptitle("CIFAR-10: Sample Image per Class", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sample_images.png"), dpi=150)
    plt.close()
    print("Saved: sample_images.png")


def plot_adversarial_accuracy(telemetry_df, plots_dir):
    """Plot adversarial accuracy vs epsilon for each attack type."""
    summary = telemetry_df.groupby(["attack_type", "epsilon"]).apply(
        lambda g: (g["true_label"] == g["adv_pred"]).mean()
    ).reset_index(name="adv_accuracy")

    fig, ax = plt.subplots(figsize=(8, 5))
    for attack_type, group in summary.groupby("attack_type"):
        group = group.sort_values("epsilon")
        ax.plot(group["epsilon"], group["adv_accuracy"],
                marker="o", linewidth=2, label=attack_type.upper())

    ax.set_xlabel("Epsilon (L∞ perturbation budget)")
    ax.set_ylabel("Adversarial Accuracy")
    ax.set_title("Model Accuracy Under Adversarial Attack")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "adversarial_accuracy.png"), dpi=150)
    plt.close()
    print("Saved: adversarial_accuracy.png")


def plot_attack_success_rate(telemetry_df, plots_dir):
    """Plot attack success rate vs epsilon."""
    summary = telemetry_df.groupby(["attack_type", "epsilon"])["flipped"].mean() \
        .reset_index(name="asr")

    fig, ax = plt.subplots(figsize=(8, 5))
    for attack_type, group in summary.groupby("attack_type"):
        group = group.sort_values("epsilon")
        ax.plot(group["epsilon"], group["asr"],
                marker="s", linewidth=2, label=attack_type.upper())

    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack Success Rate vs Perturbation Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "attack_success_rate.png"), dpi=150)
    plt.close()
    print("Saved: attack_success_rate.png")


def plot_confidence_distribution(telemetry_df, plots_dir):
    """Plot clean vs adversarial confidence distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Filter to one epsilon for clarity
    subset = telemetry_df[telemetry_df["epsilon"] == 0.04]

    axes[0].hist(subset["clean_conf"], bins=50, color="steelblue",
                 alpha=0.7, edgecolor="black", label="Clean")
    axes[0].hist(subset["adv_conf"], bins=50, color="coral",
                 alpha=0.7, edgecolor="black", label="Adversarial")
    axes[0].set_title("Confidence Distribution (ε=0.04)")
    axes[0].set_xlabel("Prediction Confidence")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Confidence drop by attack type
    conf_drop = subset.groupby("attack_type").apply(
        lambda g: g["clean_conf"].mean() - g["adv_conf"].mean()
    ).reset_index(name="conf_drop")

    axes[1].bar(conf_drop["attack_type"].str.upper(), conf_drop["conf_drop"],
                color=["steelblue", "coral"], edgecolor="black")
    axes[1].set_title("Mean Confidence Drop (ε=0.04)")
    axes[1].set_ylabel("Confidence Drop")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confidence_distribution.png"), dpi=150)
    plt.close()
    print("Saved: confidence_distribution.png")


def plot_per_class_asr_heatmap(telemetry_df, plots_dir):
    """Heatmap of attack success rate per class and epsilon."""
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    for attack_type in telemetry_df["attack_type"].unique():
        subset = telemetry_df[telemetry_df["attack_type"] == attack_type]
        pivot = subset.pivot_table(
            values="flipped", index="true_label", columns="epsilon", aggfunc="mean"
        )
        pivot.index = [class_names[i] for i in pivot.index]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                    ax=ax, vmin=0, vmax=1, linewidths=0.5)
        ax.set_title(f"Per-Class Attack Success Rate ({attack_type.upper()})")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Class")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"class_asr_heatmap_{attack_type}.png"), dpi=150
        )
        plt.close()
        print(f"Saved: class_asr_heatmap_{attack_type}.png")


def plot_correlation_heatmap(features_df, plots_dir):
    """Correlation heatmap of robustness features."""
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    corr = features_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, center=0, linewidths=0.5, square=True)
    ax.set_title("Correlation Heatmap: Robustness Features")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_correlation.png"), dpi=150)
    plt.close()
    print("Saved: feature_correlation.png")


def print_telemetry_summary(telemetry_df):
    """Print summary statistics for the telemetry dataset."""
    print("\n" + "=" * 60)
    print("TELEMETRY DATASET SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(telemetry_df)}")
    print(f"Unique images: {telemetry_df['image_id'].nunique()}")
    print(f"Attack types: {telemetry_df['attack_type'].unique().tolist()}")
    print(f"Epsilon values: {sorted(telemetry_df['epsilon'].unique().tolist())}")
    print(f"\nMissing values per column:")
    print(telemetry_df.isnull().sum().to_string())
    print(f"\nDuplicate rows: {telemetry_df.duplicated().sum()}")
    print(f"\nNumeric summary stats:")
    print(telemetry_df[["clean_conf", "adv_conf", "l_inf_norm", "flipped"]]
          .describe().round(4).to_string())
    print(f"\nSchema:")
    print(telemetry_df.dtypes.to_string())
    print(f"\nSample rows:")
    print(telemetry_df.head(5).to_string(index=False))
    print("=" * 60)


def main():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    plots_dir = config["outputs"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    # CIFAR-10 data profile plots
    data_dir = config["data"]["data_dir"]
    plot_cifar10_class_distribution(data_dir, plots_dir)
    plot_pixel_distribution(data_dir, plots_dir)
    plot_sample_images(data_dir, plots_dir)

    # Load telemetry and features
    telemetry_path = config["outputs"]["telemetry_path"]
    features_path = config["outputs"]["features_path"]

    if os.path.exists(telemetry_path):
        telemetry_df = pd.read_csv(telemetry_path)
        print_telemetry_summary(telemetry_df)
        plot_adversarial_accuracy(telemetry_df, plots_dir)
        plot_attack_success_rate(telemetry_df, plots_dir)
        plot_confidence_distribution(telemetry_df, plots_dir)
        plot_per_class_asr_heatmap(telemetry_df, plots_dir)
    else:
        print(f"Warning: Telemetry not found at {telemetry_path}. Run generate_telemetry.py first.")

    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        plot_correlation_heatmap(features_df, plots_dir)
    else:
        print(f"Warning: Features not found at {features_path}. Run feature_engineering.py first.")

    print(f"\nAll plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
