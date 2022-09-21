from typing import Iterable, Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure

ITER_STR = Iterable[str] | str
ITER_FLOAT = Iterable[float] | float
OPT_ITER_STR = Optional[ITER_STR]
OPT_ITER_FLOAT = Optional[ITER_FLOAT]

PLOT = tuple[Figure, Axes]
OPT_PLOT = Optional[PLOT]
