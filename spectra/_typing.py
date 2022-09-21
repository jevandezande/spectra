from typing import Iterable, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure

ITER_STR = Optional[Iterable[str] | str]
ITER_FLOAT = Optional[Iterable[float] | float]

OPT_PLOT = Optional[tuple[Figure, Axes]]
