from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from .tools import index_of_x, integrate, read_csvs

Self = TypeVar("Self", bound="Spectrum")


class Spectrum(ABC):
    def __init__(
        self,
        name: str,
        energies: ArrayLike,
        intensities: ArrayLike,
        units: Optional[str] = None,
        style: Optional[str] = None,
        time=None,
    ):
        energies = np.asarray(energies)
        intensities = np.asarray(intensities)
        assert len(energies.shape) == 1
        assert energies.shape == intensities.shape

        self.name = name
        self.energies = np.asarray(energies, dtype=float)
        self.intensities = np.asarray(intensities, dtype=float)
        self.units = units
        self.style = style
        self.time = time

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self) -> str:
        return repr(self)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        """
        Iterate over points in the Spectrum.

        !!!Warning, conceptually different definition between ConvSpectrum and SticksSpectrum!!!
        :yield: energy, intensity
        """
        yield from zip(self.energies, self.intensities)

    def __eq__(self, other) -> bool:
        return (
            self.name == other.name
            and self.energies.shape == other.energies.shape
            and (self.energies == other.energies).all()
            and (self.intensities == other.intensities).all()
            and self.units == other.units
            and self.style == other.style
        )

    def __len__(self) -> int:
        """
        Number of energies in the Spectrum.

        !!!Warning, conceptually different definition between ConvSpectrum and SticksSpectrum!!!
        """
        return len(self.energies)

    def __abs__(self: Self) -> Self:
        new = self.copy()
        new.name = f"|{self.name}|"
        new.intensities = abs(self.intensities)
        return new

    def __radd__(self, other: float) -> Self:
        return self.__add__(other)

    @abstractmethod
    def __add__(self, other: float) -> Self:
        pass

    @abstractmethod
    def __sub__(self, other: float) -> Self:
        pass

    def __rtruediv__(self: Self, other: float) -> Self:
        new = self.copy()
        new.name = f"{other} / {self.name}"
        new.intensities = other / new.intensities
        return new

    def __truediv__(self: Self, other: Spectrum | float) -> Self:
        intensity_divisor: np.ndarray | float
        if isinstance(other, type(self)):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot divide {self.__class__.__name__} with different units.")
            elif self.energies.shape != other.energies.shape:
                raise NotImplementedError(f"Cannot divide {self.__class__.__name__} with different shapes.")
            elif any(self.energies != other.energies):
                raise NotImplementedError(f"Cannot divide {self.__class__.__name__} with different energies")
            other_name = other.name
            intensity_divisor = other.intensities
        elif isinstance(other, Spectrum):
            raise TypeError(f"Cannot divide Spectra of different types: {type(self)} != {type(other)=}")
        else:
            other_name = f"{other}"
            intensity_divisor = other

        new = self.copy()
        new.name = f"{self.name} / {other_name}"
        new.intensities /= intensity_divisor
        return new

    def __rmul__(self: Self, other: float) -> Self:
        return self.__mul__(other)

    def __mul__(self: Self, other: Spectrum | float) -> Self:
        intensity_multiplier: np.ndarray | float
        if isinstance(other, type(self)):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot multiply {self.__class__.__name__} with different units.")
            elif self.energies.shape != other.energies.shape:
                raise NotImplementedError(f"Cannot multiply {self.__class__.__name__} with different shapes.")
            elif any(self.energies != other.energies):
                raise NotImplementedError(f"Cannot multiply {self.__class__.__name__} with different energies.")
            other_name = other.name
            intensity_multiplier = other.intensities
        elif isinstance(other, Spectrum):
            raise TypeError(f"Cannot multiply Spectra of different types: {type(self)} != {type(other)=}")
        else:
            other_name = f"{other}"
            intensity_multiplier = other

        new = self.copy()
        new.name = f"{self.name} * {other_name}"
        new.intensities *= intensity_multiplier
        return new

    def _intensities(self, energy: float, energy2: Optional[float] = None) -> np.ndarray | float:
        raise NotImplementedError()

    @property
    def min(self) -> tuple[float, float]:
        """
        Determine the min intensity and coordinate energy.

        :return: energy, min_intensity
        """
        min_idx = np.argmin(self.intensities)
        return self.energies[min_idx], self.intensities[min_idx]

    @property
    def max(self) -> tuple[float, float]:
        """
        Determine the max y and coordinate x.

        :return: x, max_y
        """
        max_idx = np.argmax(self.intensities)
        return self.energies[max_idx], self.intensities[max_idx]

    @property
    def domain(self) -> tuple[float, float]:
        """
        Domain of the Spectrum (range of x-values).

        :return: first x, last x
        """
        return self.energies[0], self.energies[-1]

    def baseline_subtracted(self: Self, val: float | bool = True) -> Self:
        """
        Make a new Spectrum with the baseline subtracted.

        :param val: amount to subtract, if None, use the lowest value.
        :return: Spectrum with the baseline subtracted.
        """
        assert val is not False

        sub_val = val if not isinstance(val, bool) else self.intensities.min()
        new = self.copy()
        new.intensities -= sub_val  # type:ignore
        return new

    def set_zero(self: Self, energy: float, energy2: Optional[float] = None) -> Self:
        """
        Set energy (or range of energies) at which intensity (or average intensity) is set to 0.

        :param energy: value at which intensity is set to zero
        :param energy: end of range (unless None)
        :return: zeroed Spectrum
        """
        delta = self._intensities(energy) if energy2 is None else np.mean(self._intensities(energy, energy2))
        if TYPE_CHECKING:
            assert isinstance(delta, float)
        return self.baseline_subtracted(delta)

    def correlation(self, other: Spectrum) -> float:
        """
        Determine the correlation between two Spectra.

        :return: correlation score in [-1, 1]
        """
        if (
            len(self.energies) != len(other.energies)
            or any(self.energies != other.energies)
            or type(self) != type(other)
        ):
            raise NotImplementedError("Cannot determine the correlation of disparate spectra.")

        return sum(self.intensities * other.intensities) / (self.norm * other.norm)

    @property
    def norm(self) -> float:
        """
        Determine the Frobenius norm of the Spectrum.
        """
        return np.linalg.norm(self.intensities)  # type: ignore

    def normed(
        self: Self,
        target: tuple[float, float] | float | str = "area",
        target_value: float = 1,
    ) -> Self:
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
                norm = integrate(self.energies, self.intensities)
            elif target == "max":
                norm = max(self.intensities)
            else:
                raise ValueError("{target=} not supported")
        else:
            # if a number
            if isinstance(target, tuple):
                try:
                    a, b = map(float, target)
                except ValueError:
                    raise ValueError(f"Could not normalize a Spectrum with {target=}")
                norm = integrate(self.energies, self.intensities, target)
            else:
                norm = self._intensities(target)  # type: ignore

        new = self.copy()
        new.intensities *= target_value / norm
        return new

    def copy(self: Self) -> Self:
        """
        Create a copy of the Spectrum.
        """
        return type(self)(
            f"{self.name}",
            np.copy(self.energies),
            np.copy(self.intensities),
            units=self.units,
            style=self.style,
            time=self.time,
        )

    def sliced(self: Self, start: Optional[float] = None, end: Optional[float] = None) -> Self:
        """
        Make a new Spectrum that is a slice of self.

        :param start: the start of the slice.
        :param end: the end of the slice.
        :return: new, sliced Spectrum.
        """
        start_i = index_of_x(start, self.energies) if start is not None else None
        end_i = index_of_x(end, self.energies) if end is not None else None

        if TYPE_CHECKING:
            assert isinstance(start_i, int)
            assert isinstance(end_i, int)

        new = self.copy()
        new.energies = new.energies[start_i:end_i]
        new.intensities = new.intensities[start_i:end_i]
        return new

    @abstractmethod
    def smoothed(self, box_pts: int | bool = True) -> Self:
        pass

    @classmethod
    def from_csvs(cls: type[Self], *inps: str, names: Optional[Iterable[str]] = None) -> list[Self]:
        """
        Read from csvs.

        :param inps: file names of the csvs
        :param names: names of the Spectra
        :return: list of Spectra
        """
        ns, energies, intensities = read_csvs(inps)
        names = ns if names is None else names
        return [cls(name, energies, intensities) for name, energies, intensities in zip(names, energies, intensities)]
