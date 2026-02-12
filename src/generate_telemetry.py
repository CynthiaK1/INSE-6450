"""
Generate adversarial evaluation telemetry.
Runs attacks on the CIFAR-10 test set and records per-sample metrics.
Outputs a CSV file with one row per (image, attack_config) pair.
"""

import os
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms

from train_model import build_resnet18
from adversarial_attacks import fgsm_attack, pgd_attack


def generate_telemetry(config):
    """Run all attacks and generate telemetry CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = build_resnet18(num_classes=config["model"]["num_classes"])
    model.load_state_dict(torch.load(
        config["model"]["checkpoint_path"], map_location=device, weights_only=True
    ))
    model.to(device)
    model.eval()

    # Load test data (same normalization as training)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_set = datasets.CIFAR10(
        root=config["data"]["data_dir"], train=False,
        download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"]
    )

    # Collect telemetry records
    records = []
    image_id_offset = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Clean predictions
        with torch.no_grad():
            clean_outputs = model(images)
            clean_probs = F.softmax(clean_outputs, dim=1)
            clean_confs, clean_preds = clean_probs.max(dim=1)

        # Run each attack configuration
        for attack_cfg in config["attacks"]:
            attack_type = attack_cfg["type"]
            for epsilon in attack_cfg["epsilons"]:
                # Generate adversarial examples
                if attack_type == "fgsm":
                    adv_images = fgsm_attack(model, images, labels, epsilon, device)
                elif attack_type == "pgd":
                    adv_images = pgd_attack(
                        model, images, labels, epsilon,
                        alpha=attack_cfg.get("pgd_alpha", 0.005),
                        steps=attack_cfg.get("pgd_steps", 10),
                        device=device
                    )
                else:
                    raise ValueError(f"Unknown attack type: {attack_type}")

                # Adversarial predictions
                with torch.no_grad():
                    adv_outputs = model(adv_images)
                    adv_probs = F.softmax(adv_outputs, dim=1)
                    adv_confs, adv_preds = adv_probs.max(dim=1)

                # Compute per-sample L-inf norm of actual perturbation
                l_inf = (adv_images - images).abs().view(batch_size, -1).max(dim=1)[0]

                # Record telemetry
                for i in range(batch_size):
                    records.append({
                        "image_id": image_id_offset + i,
                        "true_label": labels[i].item(),
                        "clean_pred": clean_preds[i].item(),
                        "clean_conf": round(clean_confs[i].item(), 6),
                        "adv_pred": adv_preds[i].item(),
                        "adv_conf": round(adv_confs[i].item(), 6),
                        "flipped": int(clean_preds[i].item() != adv_preds[i].item()),
                        "attack_type": attack_type,
                        "epsilon": epsilon,
                        "l_inf_norm": round(l_inf[i].item(), 6),
                    })

        image_id_offset += batch_size
        if image_id_offset % 2000 == 0:
            print(f"Processed {image_id_offset}/10000 images")

    # Save telemetry
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(config["outputs"]["telemetry_path"]), exist_ok=True)
    df.to_csv(config["outputs"]["telemetry_path"], index=False)
    print(f"Telemetry saved to {config['outputs']['telemetry_path']} ({len(df)} records)")
    return df


def main():
    with open("configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_telemetry(config)


if __name__ == "__main__":
    main()
