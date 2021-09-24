from __future__ import annotations

from typing import Iterable, Optional, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

ITER_STR = Optional[Union[Iterable[str], str]]
ITER_FLOAT = Optional[Union[Iterable[float], float]]

OPT_PLOT = Optional[tuple[Figure, Axes]]
