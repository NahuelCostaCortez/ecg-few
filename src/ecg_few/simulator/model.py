"""Core waveform primitives used by the simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianAtom:
    """Single Gaussian atom: a * exp(-((t-mu)^2) / (2*sigma^2))."""

    name: str
    a: float
    mu: float
    sigma: float

    def evaluate(self, t_ms: np.ndarray) -> np.ndarray:
        return self.a * np.exp(-((t_ms - self.mu) ** 2) / (2.0 * self.sigma**2))

    def as_dict(self) -> dict[str, float]:
        return {
            "name": self.name,
            "a": float(self.a),
            "mu_ms": float(self.mu),
            "sigma_ms": float(self.sigma),
        }


__all__ = ["GaussianAtom"]