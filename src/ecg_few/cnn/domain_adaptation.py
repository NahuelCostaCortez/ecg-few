"""Unsupervised domain adaptation helpers for CNN QRS detectors."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from ecg_few.cnn.data import PreprocessConfig, UnlabeledECGImageDataset
from ecg_few.loocv import BrugadaImageRow


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, gradients: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * gradients, None


class DomainClassifier(nn.Module):
    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class SimCLRProjectionHead(nn.Module):
    def __init__(self, feature_dim: int = 512, projection_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def resnet_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def resnet_logits_from_features(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return model.fc(features)


def make_unlabeled_loader(
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    preprocess: PreprocessConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        UnlabeledECGImageDataset(rows, dataset_root, preprocess),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def cycle_batches(loader: DataLoader[torch.Tensor]) -> Iterable[torch.Tensor]:
    while True:
        yield from loader


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    if source_features.shape[0] < 2 or target_features.shape[0] < 2:
        return source_features.new_tensor(0.0)
    source = source_features - source_features.mean(dim=0, keepdim=True)
    target = target_features - target_features.mean(dim=0, keepdim=True)
    source_cov = source.t().matmul(source) / float(source.shape[0] - 1)
    target_cov = target.t().matmul(target) / float(target.shape[0] - 1)
    feature_dim = source_features.shape[1]
    return (source_cov - target_cov).pow(2).sum() / float(4 * feature_dim * feature_dim)


def mmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    kernel_scales: Sequence[float],
) -> torch.Tensor:
    if source_features.numel() == 0 or target_features.numel() == 0:
        return source_features.new_tensor(0.0)
    source = F.normalize(source_features, dim=1)
    target = F.normalize(target_features, dim=1)
    xx = _rbf_kernel(source, source, kernel_scales)
    yy = _rbf_kernel(target, target, kernel_scales)
    xy = _rbf_kernel(source, target, kernel_scales)
    return xx.mean() + yy.mean() - 2.0 * xy.mean()


def _rbf_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    kernel_scales: Sequence[float],
) -> torch.Tensor:
    distance = torch.cdist(left, right).pow(2)
    kernels = [torch.exp(-distance / (2.0 * float(scale) ** 2)) for scale in kernel_scales]
    return torch.stack(kernels, dim=0).mean(dim=0)


def domain_adaptation_weight(epoch: int, total_epochs: int, base_weight: float) -> float:
    if base_weight <= 0:
        return 0.0
    progress = min(1.0, max(0.0, float(epoch) / max(1.0, float(total_epochs))))
    # DANN-style smooth schedule; also works as a conservative ramp for CORAL/MMD.
    return float(base_weight) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def build_ssl_encoder(*, resnet_weights: str) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if resnet_weights == "default" else None
    encoder = models.resnet18(weights=weights)
    encoder.fc = nn.Identity()
    return encoder


def copy_encoder_weights(supervised_model: nn.Module, encoder: nn.Module) -> None:
    encoder_state = {
        key: value
        for key, value in encoder.state_dict().items()
        if not key.startswith("fc.")
    }
    supervised_state = supervised_model.state_dict()
    supervised_state.update(
        {
            key: value
            for key, value in encoder_state.items()
            if key in supervised_state and supervised_state[key].shape == value.shape
        }
    )
    supervised_model.load_state_dict(supervised_state)


def pretrain_encoder_simclr(
    *,
    target_rows: Sequence[BrugadaImageRow],
    target_dataset_root: Path,
    preprocess: PreprocessConfig,
    run_dir: Path,
    seed: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    resnet_weights: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    projection_dim: int,
    temperature: float,
) -> tuple[nn.Module | None, dict[str, object]]:
    if epochs <= 0 or not target_rows:
        return None, {"ssl_pretrain_epochs": 0, "ssl_pretrain_seconds": 0.0}
    random.seed(seed)
    torch.manual_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    loader = make_unlabeled_loader(
        target_rows,
        dataset_root=target_dataset_root,
        preprocess=preprocess,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    encoder = build_ssl_encoder(resnet_weights=resnet_weights).to(device)
    projector = SimCLRProjectionHead(projection_dim=projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        encoder.train()
        projector.train()
        total_loss = 0.0
        total = 0
        for images in loader:
            images = images.to(device, non_blocking=True)
            if images.shape[0] < 2:
                continue
            view_a = augment_batch(images, image_size=image_size)
            view_b = augment_batch(images, image_size=image_size)
            features_a = projector(encoder(view_a))
            features_b = projector(encoder(view_b))
            loss = nt_xent_loss(features_a, features_b, temperature=temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(images.shape[0])
            total += int(images.shape[0])
        history.append({"epoch": epoch, "ssl_loss": total_loss / max(1, total)})
    seconds = time.perf_counter() - started
    torch.save(encoder.state_dict(), run_dir / "simclr_encoder.pt")
    (run_dir / "simclr_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        encoder,
        {
            "ssl_pretrain_epochs": epochs,
            "ssl_pretrain_seconds": seconds,
            "ssl_pretrain_patients": len({row.patient_id for row in target_rows}),
            "ssl_pretrain_images": len(target_rows),
            "ssl_projection_dim": projection_dim,
            "ssl_temperature": temperature,
            "ssl_final_loss": history[-1]["ssl_loss"] if history else "",
        },
    )


def augment_batch(images: torch.Tensor, *, image_size: int) -> torch.Tensor:
    augmented = images.clone()
    batch_size = int(augmented.shape[0])
    scale = torch.empty(batch_size, 1, 1, 1, device=augmented.device).uniform_(0.85, 1.15)
    shift = torch.empty(batch_size, 1, 1, 1, device=augmented.device).uniform_(-0.08, 0.08)
    augmented = augmented * scale + shift
    noise = torch.randn_like(augmented) * 0.03
    augmented = augmented + noise
    for index in range(batch_size):
        max_shift = max(1, image_size // 32)
        dy = random.randint(-max_shift, max_shift)
        dx = random.randint(-max_shift, max_shift)
        augmented[index] = torch.roll(augmented[index], shifts=(dy, dx), dims=(-2, -1))
    if random.random() < 0.5:
        kernel = torch.ones((3, 1, 3, 3), device=augmented.device, dtype=augmented.dtype) / 9.0
        augmented = F.conv2d(augmented, kernel, padding=1, groups=3)
    return augmented


def nt_xent_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    features = torch.cat([features_a, features_b], dim=0)
    logits = features.matmul(features.t()) / max(float(temperature), 1e-6)
    batch_size = int(features_a.shape[0])
    mask = torch.eye(2 * batch_size, device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))
    labels = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=features.device),
            torch.arange(0, batch_size, device=features.device),
        ]
    )
    return F.cross_entropy(logits, labels)
