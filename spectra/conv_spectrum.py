from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Iterable

import numpy as np
from scipy import signal

from ._abc_spectrum import Spectrum
from .tools import index_of_x, integrate, read_csvs, smooth_curve, y_at_x


class ConvSpectrum(Spectrum):
    def __init__(
        self,
        name: str,
        xs: np.ndarray,
        ys: np.ndarray,
        units: str = None,
        style: str = None,
        time=None,
    ) -> None:
        """
        A ConvSpectrum is a collection of intensities (ys) at various frequencies or energies (xs).
        It is a convetional spectrum, but can also be interpretted as a convolved spectrum.

        :param name: name of the ConvSpectrum
        :param xs: x-values
        :param ys: intensities
        :param units: units for the x-values
        :param style: style of ConvSpectrum (e.g. IR, XRD, etc.)
        """
        super().__init__(name, xs, ys, units, style, time)

    def __iter__(self) -> Generator[tuple[float, float], None, None]:
        """
        Iterate over points in the ConvSpectrum.

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
        Number of points in the ConvSpectrum.
        """
        return len(self.xs)

    def __rsub__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        return self.__class__(f"{self.name}", np.copy(self.xs), other - self.ys)  # type: ignore

    def __sub__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different units.")
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different shapes.")
            elif any(self.xs != other.xs):
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different x-values.")
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

    def __radd__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        return self.__add__(other)

    def __add__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different units.")
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different shapes.")
            elif any(self.xs != other.xs):
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different x-values.")
            return self.__class__(f"{self.name} + {other.name}", np.copy(self.xs), self.ys + other.ys)
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            self.ys + other,
            units=self.units,
            style=self.style,
        )

    def __abs__(self) -> ConvSpectrum:
        return self.__class__(
            f"|{self.name}|",
            np.copy(self.xs),
            abs(self.ys),
            units=self.units,
            style=self.style,
        )

    def __rtruediv__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        return self.__class__(f"{other}/{self.name}", np.copy(self.xs), other / self.ys)  # type: ignore

    def __truediv__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot divide {self.__class__.__name__} with different units.")
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(f"Cannot divide {self.__class__.__name__} with different shapes.")
            elif any(self.xs != other.xs):
                raise NotImplementedError(f"Cannot divide {self.__class__.__name__} with different x-values.")
            return self.__class__(f"{self.name} / {other.name}", np.copy(self.xs), self.ys / other.ys)
        return self.__class__(f"{self.name}", np.copy(self.xs), self.ys / other)

    def __rmul__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        return self.__mul__(other)

    def __mul__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot multiply {self.__class__.__name__} with different units.")
            elif self.xs.shape != other.xs.shape:
                raise NotImplementedError(f"Cannot multiply {self.__class__.__name__} with different shapes.")
            elif any(self.xs != other.xs):
                raise NotImplementedError(f"Cannot multiply {self.__class__.__name__} with different x-values.")
            return self.__class__(f"{self.name} * {other.name}", np.copy(self.xs), self.ys * other.ys)
        return self.__class__(f"{self.name}", np.copy(self.xs), self.ys * other)

    def _ys(self, x: float, x2: float = None) -> np.ndarray | float:
        """
        Directly access the y-value(s) at x to x2.

        :param x: x-value at which to evaluate (or start).
        :param x2: x-value at which to end, if None, only the value at x is returned.
        :return: y or np.ndarray of ys..
        """
        if x2 is None:
            return y_at_x(x, self.xs, self.ys)
        return self.ys[index_of_x(x, self.xs) : index_of_x(x2, self.xs)]  # type: ignore

    def copy(self) -> ConvSpectrum:
        """
        Create a copy of the ConvSpectrum.
        """
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            np.copy(self.ys),
            units=self.units,
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
        Domain of the ConvSpectrum (range of x-values).

        :return: first x, last x
        """
        return self.xs[0], self.xs[-1]

    def correlation(self, other: ConvSpectrum) -> float:
        """
        Determine the correlation between two Spectra.

        :return: correlation score in [-1, 1]
        """
        if len(self.xs) != len(other.xs) or any(self.xs != other.xs):
            raise NotImplementedError("Cannot determine the correlation of Spectra with different x-values.")

        return sum(self.ys * other.ys) / (self.norm * other.norm)

    def smoothed(self, box_pts: int | bool = True) -> ConvSpectrum:
        """
        Make a smoothed version of the ConvSpectrum.

        :param box_pts: number of data points to convolve, if True, use 3
        :return: smoothed ConvSpectrum
        """
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            smooth_curve(self.ys, box_pts),
            units=self.units,
            style=self.style,
        )

    def baseline_subtracted(self, val: float | bool = True) -> ConvSpectrum:
        """
        Make a new ConvSpectrum with the baseline subtracted.

        :param val: amount to subtract, if None, use the lowest value.
        :return: ConvSpectrum with the baseline subtracted.
        """
        assert not (val is False)

        sub_val = val if not isinstance(val, bool) else self.ys.min()
        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
            self.ys - sub_val,  # type: ignore
            units=self.units,
            style=self.style,
        )

    def set_zero(self, x: float, x2: float = None) -> ConvSpectrum:
        """
        Set x (or range of x) at which y (or y average) is set to 0.

        :param x: value at which y is set to zero
        :param x2: end of range (unless None)
        :return: zeroed ConvSpectrum
        """
        delta = self._ys(x) if x2 is None else np.mean(self._ys(x, x2))
        if TYPE_CHECKING:
            assert isinstance(delta, float)
        return self.baseline_subtracted(delta)

    def sliced(self, start: float = None, end: float = None) -> ConvSpectrum:
        """
        Make a new ConvSpectrum that is a slice of self.

        :param start: the start of the slice.
        :param end: the end of the slice.
        :return: new, sliced ConvSpectrum.
        """
        xs, ys = self.xs, self.ys

        start_i = index_of_x(start, xs) if start is not None else None
        end_i = index_of_x(end, xs) if end is not None else None

        if TYPE_CHECKING:
            assert isinstance(start_i, int)
            assert isinstance(end_i, int)

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
        Determine the Frobenius norm of the ConvSpectrum.
        """
        return np.linalg.norm(self.ys)

    def normed(
        self,
        target: tuple[float, float] | float | str = "area",
        target_value: float = 1,
    ) -> ConvSpectrum:
        """
        Make a normalized ConvSpectrum.

        :param target:
            'area' - normalize using total area
            'max' - normalize based on max value
            x-value - normalize based on the y-value at this x-value
            (start, end) - normalize based on integration from start to end
        :param target_value: what to normalize the target to
        :return: normalized ConvSpectrum
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
                    a, b = map(float, target)
                except ValueError:
                    raise ValueError(f"Could not normalize a ConvSpectrum with {target=}")
                norm = integrate(self.xs, self.ys, target)
            else:
                norm = self._ys(target)  # type: ignore

        return self.__class__(
            f"{self.name}",
            np.copy(self.xs),
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
    ) -> tuple[np.ndarray, np.ndarray]:
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
