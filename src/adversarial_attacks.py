"""
Adversarial attack implementations: FGSM and PGD.
Operates on normalized CIFAR-10 inputs.
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, device):
    """
    Fast Gradient Sign Method (FGSM).

    Args:
        model: Target classifier
        images: Clean input tensor (batch, C, H, W), normalized
        labels: True labels
        epsilon: Perturbation budget (L-inf)
        device: torch device

    Returns:
        adv_images: Adversarial examples
    """
    images = images.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    # FGSM perturbation
    perturbation = epsilon * images.grad.sign()
    adv_images = images + perturbation

    return adv_images.detach()


def pgd_attack(model, images, labels, epsilon, alpha, steps, device):
    """
    Projected Gradient Descent (PGD) attack.

    Args:
        model: Target classifier
        images: Clean input tensor
        labels: True labels
        epsilon: Perturbation budget (L-inf)
        alpha: Step size per iteration
        steps: Number of PGD iterations
        device: torch device

    Returns:
        adv_images: Adversarial examples
    """
    adv_images = images.clone().detach().to(device)
    original_images = images.clone().detach().to(device)
    labels = labels.to(device)

    # Random start within epsilon ball
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)

    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        # PGD step
        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()

        # Project back into epsilon ball
        perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
        adv_images = original_images + perturbation

    return adv_images.detach()
