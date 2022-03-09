import numpy as np
from numpy.typing import ArrayLike


def gaussian(energy: float, width: float, xs: ArrayLike) -> np.ndarray:
    xs = np.asarray(xs)
    return np.exp(-((xs - energy) ** 2) / (2 * width**2))
