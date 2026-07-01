"""CNN dataset helpers for QRS-finding LOOCV."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ecg_few.findings import LABEL_NAMES
from ecg_few.loocv import BrugadaImageRow


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int
    mean: float
    std: float


class ECGImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: Sequence[BrugadaImageRow],
        dataset_root: Path,
        preprocess: PreprocessConfig,
    ) -> None:
        self.rows = list(rows)
        self.dataset_root = dataset_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = load_image_tensor(
            self.dataset_root / row.image_path,
            image_size=self.preprocess.image_size,
            mean=self.preprocess.mean,
            std=self.preprocess.std,
        )
        if not row.has_qrs_labels:
            raise ValueError(
                f"Row {row.patient_id}/{row.lead} has no QRS labels and cannot train "
                "the multi-label detector."
            )
        label = torch.tensor(
            [float(row.findings[label_name]) for label_name in LABEL_NAMES],
            dtype=torch.float32,
        )
        return image, label


class UnlabeledECGImageDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        rows: Sequence[BrugadaImageRow],
        dataset_root: Path,
        preprocess: PreprocessConfig,
    ) -> None:
        self.rows = list(rows)
        self.dataset_root = dataset_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        return load_image_tensor(
            self.dataset_root / row.image_path,
            image_size=self.preprocess.image_size,
            mean=self.preprocess.mean,
            std=self.preprocess.std,
        )


def load_raw_image(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        grayscale = image.convert("L")
        resized = grayscale.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0


def load_image_tensor(image_path: Path, image_size: int, mean: float, std: float) -> torch.Tensor:
    array = load_raw_image(image_path, image_size)
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    mean_tensor = torch.full((3, 1, 1), float(mean), dtype=tensor.dtype)
    std_tensor = torch.full((3, 1, 1), max(float(std), 1e-6), dtype=tensor.dtype)
    return (tensor - mean_tensor) / std_tensor


def compute_preprocess_config(
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    image_size: int,
) -> PreprocessConfig:
    if not rows:
        return PreprocessConfig(image_size=image_size, mean=0.5, std=0.25)
    pixels: list[np.ndarray] = []
    for row in rows:
        pixels.append(load_raw_image(dataset_root / row.image_path, image_size).reshape(-1))
    values = np.concatenate(pixels)
    return PreprocessConfig(
        image_size=image_size,
        mean=float(values.mean()),
        std=float(values.std() or 1.0),
    )
