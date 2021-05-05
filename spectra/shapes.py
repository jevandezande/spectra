import numpy as np


def gaussian(energy: float, width: float, xs: np.ndarray) -> np.ndarray:
    return np.exp(-((xs - energy) ** 2) / (2 * width ** 2))
