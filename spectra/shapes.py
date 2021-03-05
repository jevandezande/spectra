from __future__ import annotations

import numpy as np


def gaussian(energy: float, intensity: float, width: float, xs: np.ndarray) -> np.ndarray:
    return intensity * np.exp(-((xs - energy) ** 2) / (2 * width ** 2))


def lorentzian(energy: float, intensity: float, gamma: float, xs: np.ndarray) -> np.ndarray:
    return intensity / (np.pi * gamma * (1 + ((xs - energy) / gamma) ** 2))


def cauchy(energy: float, intensity: float, gamma: float, xs: np.ndarray) -> np.ndarray:
    return lorentzian(energy, intensity, 1, xs)
