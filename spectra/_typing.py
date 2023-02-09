from typing import Iterable

from matplotlib.axes import Axes
from matplotlib.figure import Figure

ITER_STR = Iterable[str] | str
ITER_FLOAT = Iterable[float] | float
OPT_ITER_STR = ITER_STR | None
OPT_ITER_FLOAT = ITER_FLOAT | None

PLOT = tuple[Figure, Axes]
OPT_PLOT = PLOT | None
