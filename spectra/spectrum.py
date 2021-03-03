from __future__ import annotations

import numpy as np

from scipy import signal

from .tools import index_of_x, integrate, read_csvs, smooth_curve, y_at_x
from typing import Generator, Iterable


class Spectrum:
    def __init__(
        self,
        name: str,
        xs: np.array,
        ys: np.array,
        units: str = "",
        style: str = None,
        time=None,
    ) -> None:
        """
        A Spectrum is a collection of intensities (ys) at various frequencies or energies (xs).

        :param name: name of the Spectrum
        :param xs: x-values
        :param ys: intensities
        :param units: units for the x-values
        :param style: style of Spectrum (e.g. IR, XRD, etc.)
        """
        assert len(xs.shape) == 1
        assert xs.shape == ys.shape

        self.name = name
        self.xs = xs
        self.ys = ys
        self.units = units
        self.style = style
        self.time = time

    def __iter__(self) -> Generator[tuple[float, float], None, None]:
        """
        Iterate over points in the Spectrum.

        :yield: x, y
        """
        yield from zip(self.xs, self.ys)

    def __eq__(self, other) -> bool:
        return (
            self.name == other.name
            and self.xs.shape == other.xs.shape
            and (self.xs == other.xs).all()
            and (self.ys == other.ys).all()
            and self.units == other.units
            and self.style == other.style
        )

    def __len__(self) -> int:
        """
        Number of points in the Spectrum.
        """
        return len(self.xs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self) -> str:
        return repr(self)

    def __rsub__(self, other: Spectrum | float) -> Spectrum:
        return self.__class__(f"{self.name}", np.copy(self.xs), other - self.ys)

    def __sub__(self, other: Spectrum | float) -> Spectrum:
        if isinstance(other, Spectrum):
            if self.units != other.units:
                raise NotImplementedError(
                    f"Cannot subtract {self.__class__.__name__} with different units."
                )
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(
                    f"Cannot subtract {self.__class__.__name__} with different shapes."
                )
            elif any(self.xs != other.xs):
                raise NotImplementedError(
                    f"Cannot subtract {self.__class__.__name__} with different x-values."
                )
            return self.__class__(
                f"{self.name} – {other.name}",
                np.copy(self.xs),
                self.ys - other.ys,
                units=self.units,
                style=self.style,
            )
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            self.ys - other,
            units=self.units,
            style=self.style,
        )

    def __radd__(self, other: Spectrum | float) -> Spectrum:
        return self.__add__(other)

    def __add__(self, other: Spectrum | float) -> Spectrum:
        if isinstance(other, Spectrum):
            if self.units != other.units:
                raise NotImplementedError(
                    f"Cannot add {self.__class__.__name__} with different units."
                )
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(
                    f"Cannot add {self.__class__.__name__} with different shapes."
                )
            elif any(self.xs != other.xs):
                raise NotImplementedError(
                    f"Cannot add {self.__class__.__name__} with different x-values."
                )
            return self.__class__(
                f"{self.name} + {other.name}", np.copy(self.xs), self.ys + other.ys
            )
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            self.ys + other,
            units=self.units,
            style=self.style,
        )

    def __abs__(self) -> Spectrum:
        return self.__class__(
            f"|{self.name}|",
            np.copy(self.xs),
            abs(self.ys),
            units=self.units,
            style=self.style,
        )

    def __rtruediv__(self, other: Spectrum | float) -> Spectrum:
        return self.__class__(f"{other}/{self.name}", np.copy(self.xs), other / self.ys)

    def __truediv__(self, other: Spectrum | float) -> Spectrum:
        if isinstance(other, Spectrum):
            if self.units != other.units:
                raise NotImplementedError(
                    f"Cannot divide {self.__class__.__name__} with different units."
                )
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(
                    f"Cannot divide {self.__class__.__name__} with different shapes."
                )
            elif any(self.xs != other.xs):
                raise NotImplementedError(
                    f"Cannot divide {self.__class__.__name__} with different x-values."
                )
            return self.__class__(
                f"{self.name} / {other.name}", np.copy(self.xs), self.ys / other.ys
            )
        return self.__class__(f"{self.name}", np.copy(self.xs), self.ys / other)

    def __rmul__(self, other: Spectrum | float) -> Spectrum:
        return self.__mul__(other)

    def __mul__(self, other: Spectrum | float) -> Spectrum:
        if isinstance(other, Spectrum):
            if self.units != other.units:
                raise NotImplementedError(
                    f"Cannot multiply {self.__class__.__name__} with different units."
                )
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(
                    f"Cannot multiply {self.__class__.__name__} with different shapes."
                )
            elif any(self.xs != other.xs):
                raise NotImplementedError(
                    f"Cannot multiply {self.__class__.__name__} with different x-values."
                )
            return self.__class__(
                f"{self.name} * {other.name}", np.copy(self.xs), self.ys * other.ys
            )
        return self.__class__(f"{self.name}", np.copy(self.xs), self.ys * other)

    def _ys(self, x: float, x2: float = None) -> np.array | float:
        """
        Directly access the y-value(s) at x to x2.

        :param x: x-value at which to evaluate (or start).
        :param x2: x-value at which to end, if None, only the value at x is returned.
        :return: y or np.array of ys..
        """
        if x2 is None:
            return y_at_x(x, self.xs, self.ys)
        return self.ys[index_of_x(x, self.xs) : index_of_x(x2, self.xs)]

    def copy(self) -> Spectrum:
        """
        Create a copy of the Spectrum.
        """
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            np.copy(self.ys),
            units=f"{self.units}",
            style=self.style,
        )

    @property
    def min(self) -> tuple[float, float]:
        """
        Determine the min y and coordinate x.

        :return: x, min_y
        """
        min_idx = np.argmin(self.ys)
        return self.xs[min_idx], self.ys[min_idx]

    @property
    def max(self) -> tuple[float, float]:
        """
        Determine the max y and coordinate x.

        :return: x, max_y
        """
        max_idx = np.argmax(self.ys)
        return self.xs[max_idx], self.ys[max_idx]

    @property
    def domain(self) -> tuple[float, float]:
        """
        Domain of the Spectrum (range of x-values).

        :return: first x, last x
        """
        return self.xs[0], self.xs[-1]

    def correlation(self, other: Spectrum) -> float:
        """
        Determine the correlation between two Spectra.

        :return: correlation score in [-1, 1]
        """
        if len(self.xs) != len(other.xs) or any(self.xs != other.xs):
            raise NotImplementedError(
                "Cannot determine the correlation of Spectra with different x-values."
            )

        return sum(self.ys * other.ys) / (self.norm * other.norm)

    def smoothed(self, box_pts: int | bool = True) -> Spectrum:
        """
        Make a smoothed version of the Spectrum.

        :param box_pts: number of data points to convolve, if True, use 3
        :return: smoothed Spectrum
        """
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            smooth_curve(self.ys, box_pts),
            units=self.units,
            style=self.style,
        )

    def baseline_subtracted(self, val: float = None) -> Spectrum:
        """
        Make a new Spectrum with the baseline subtracted.

        :param val: amount to subtract, if None, use the lowest value.
        :return: Spectrum with the baseline subtracted.
        """
        if val is None:
            val = self.ys.min()
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            self.ys - val,
            units=self.units,
            style=self.style,
        )

    def set_zero(self, x: float, x2: float = None) -> Spectrum:
        """
        Set x (or range of x) at which y (or y average) is set to 0.

        :param x: value at which y is set to zero
        :param x2: end of range (unless None)
        :return: zeroed Spectrum
        """
        if x2 is None:
            delta = self._ys(x)
        else:
            delta = np.mean(self._ys(x, x2))

        return self.baseline_subtracted(delta)

    def sliced(self, start: float = None, end: float = None) -> Spectrum:
        """
        Make a new Spectrum that is a slice of self.

        :param start: the start of the slice.
        :param end: the end of the slice.
        :return: new, sliced Spectrum.
        """
        xs, ys = self.xs, self.ys

        start_i = index_of_x(start, xs) if start is not None else None
        end_i = index_of_x(end, xs) if end is not None else None

        return self.__class__(
            f"{self.name}",
            xs[start_i:end_i],
            ys[start_i:end_i],
            units=self.units,
            style=self.style,
        )

    @property
    def norm(self) -> float:
        """
        Determine the Frobenius norm of the Spectrum.
        """
        return np.linalg.norm(self.ys)

    def normed(
        self,
        target: tuple[float, float] | float | str = "area",
        target_value: float = 1,
    ) -> Spectrum:
        """
        Make a normalized Spectrum.

        :param target:
            'area' - normalize using total area
            'max' - normalize based on max value
            x-value - normalize based on the y-value at this x-value
            (start, end) - normalize based on integration from start to end
        :param target_value: what to normalize the target to
        :return: normalized Spectrum
        """
        if isinstance(target, str):
            if target == "area":
                norm = integrate(self.xs, self.ys)
            elif target == "max":
                norm = max(self.ys)
            else:
                raise ValueError("{target=} not supported")
        else:
            # if a number
            if isinstance(target, tuple):
                try:
                    a, b = target
                    a, b = float(a), float(b)
                except ValueError:
                    raise ValueError(f"Could not normalize a Spectrum with {target=}")
                norm = integrate(self.xs, self.ys, target)
            else:
                norm = self._ys(target)

        return self.__class__(
            f"{self.name}",
            self.xs[:],
            self.ys / norm * target_value,
            units=self.units,
            style=self.style,
        )

    def peaks(
        self,
        indices: bool = False,
        height: float = None,
        threshold: float = None,
        distance: float = None,
        prominence: float = None,
        width: float = None,
        wlen: float = None,
        rel_height: float = 0.5,
        plateau_size: float = None,
    ) -> tuple[np.array, np.array]:
        """
        Find the indices of peaks.

        Note: Utilizes scipy.signal.find_peaks and the parameters therein.

        :param indices: return peak indices instead of x-values
        :return: peak x-values (or peak indices if indices == True), properties
        """
        peaks, properties = signal.find_peaks(
            self.ys,
            height,
            threshold,
            distance,
            prominence,
            width,
            wlen,
            rel_height,
            plateau_size,
        )
        if indices:
            return peaks, properties
        return self.xs[peaks], properties


def spectra_from_csvs(*inps: str, names: Iterable[str] = None) -> list[Spectrum]:
    """
    Read from csvs.

    :param inps: file names of the csvs
    :param names: names of the Spectra
    :return: list of Spectra
    """
    ns, x_vals, y_vals = read_csvs(inps)
    if not names:
        names = ns
    return [Spectrum(name, xs, ys) for name, xs, ys in zip(names, x_vals, y_vals)]
