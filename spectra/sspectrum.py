from __future__ import annotations

from typing import TYPE_CHECKING, Generator, Iterable

import numpy as np

from .shapes import gaussian
from .spectrum import Spectrum
from .tools import read_csvs


class SSpectrum:
    def __init__(
        self,
        name: str,
        energies: Iterable,
        intensities: Iterable,
        units: str = None,
        style: str = None,
        y_shift=0,
        time=None,
    ):
        """
        A SSpectrum is a collection of intensities at various energies (or frequencies).
        These are convolved with a shape to produce a full spectrum.

        :param name: name of the SSpectrum
        :param energies: energies of transitions
        :param intensities: intensities of transitions
        :param units: units for the energies
        :param style: style of SSpectrum (e.g. IR, XRD, etc.)
        :param y_shift: global shift of intensities
        :param time: timestamp
        """
        self.name = name
        self.energies = np.array(energies)
        self.intensities = np.array(intensities)
        assert len(self.energies) == len(self.intensities)
        self.units = units
        self.style = style
        self.y_shift = y_shift
        self.time = time

    def __iter__(self) -> Generator[tuple[float, float], None, None]:
        """
        Iterate over points in the SSpectrum.

        :yield: energy, intensity
        !!!Warning, different definition than Spectrum!!!
        """
        yield from zip(self.energies, self.intensities)

    def __eq__(self, other) -> bool:
        """
        !!!Warning, different definition than Spectrum!!!
        """
        return (
            self.name == other.name
            and (self.energies == other.energies).all()
            and (self.intensities == other.intensities).all()
            and self.units == other.units
            and self.style == other.style
        )

    def __len__(self) -> int:
        """
        Number of transitions in the SSpectrum.

        !!!Warning, different definition than Spectrum!!!
        """
        return len(self.energies)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self) -> str:
        return repr(self)

    def __rsub__(self, other) -> SSpectrum:
        """
        !!!Warning, different definition than Spectrum!!!
        """
        return -other + self

    def __sub__(self, other) -> SSpectrum:
        """
        !!!Warning, different definition than Spectrum!!!
        """
        if type(self) == type(other):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different units.")
            return self.__class__(
                f"{self.name} – {other.name}",
                np.concatenate((self.energies, other.energies)),
                np.concatenate((self.intensities, -other.intensities)),
                units=self.units,
                style=self.style,
                y_shift=self.y_shift,
            )

        new = self.copy()
        new.y_shift -= other
        return new

    def __radd__(self, other) -> SSpectrum:
        return self.__add__(other)

    def __add__(self, other) -> SSpectrum:
        if type(self) == type(other):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different units.")
            return self.__class__(
                f"{self.name} + {other.name}",
                np.concatenate((self.energies, other.energies)),
                np.concatenate((self.intensities, other.intensities)),
                units=self.units,
                style=self.style,
                y_shift=self.y_shift,
            )
        new = self.copy()
        new.y_shift += other
        return new

    def __abs__(self) -> SSpectrum:
        new = self.copy()
        new.name = f"|{self.name}|"
        new.intensities = abs(self.intensities)
        return new

    def __rtruediv__(self, other) -> SSpectrum:
        raise NotImplementedError()

    def __truediv__(self, other) -> SSpectrum:
        raise NotImplementedError()

    def __rmul__(self, other) -> SSpectrum:
        raise NotImplementedError()

    def __mul__(self, other) -> SSpectrum:
        raise NotImplementedError()

    def copy(self) -> SSpectrum:
        """
        Create a copy of the SSpectrum.

        :return: duplicate SSpectrum
        """
        return self.__class__(
            self.name,
            self.energies.copy(),
            self.intensities.copy(),
            self.units,
            self.style,
            self.y_shift,
            self.time,
        )

    @property
    def min(self) -> tuple[float, float]:
        """
        Determine the min intensity coordinate energy.

        :return: x, min_y
        """
        min_idx = np.argmin(self.intensities)
        return self.energies[min_idx], self.intensities[min_idx]

    @property
    def max(self) -> tuple[float, float]:
        """
        Determine the max intensity and coordinate energy.

        :return: x, max_intensity
        """
        max_idx = np.argmax(self.intensities)
        return self.energies[max_idx], self.intensities[max_idx]

    @property
    def domain(self) -> tuple[float, float]:
        """
        Domain of the SSpectrum (range of energies).

        :return: min energy, max energy
        """
        return float(self.energies.min()), float(self.energies.max())

    def correlation(self, other) -> float:
        """
        Determine the correlation between two SSpectra.

        :return: correlation score in [-1, 1]
        """
        raise NotImplementedError()

    def smoothed(self, box_pts=True) -> SSpectrum:
        """
        Return a smoothed version of the SSpectrum.

        :param box_pts: number of data points to convolve, if True, use 3
        :return: smoothed SSpectrum
        """
        raise NotImplementedError()

    def baseline_subtracted(self, val=None) -> SSpectrum:
        """
        Return a new SSpectrum with the baseline subtracted.

        :param val: amount to subtract, if None, use the lowest value.
        :return: SSpectrum with the baseline subtracted.
        """
        if val is None:
            val = self.intensities.min()
        return self.__class__(
            self.name,
            self.energies.copy(),
            self.intensities - val,
            self.units,
            self.style,
            self.y_shift,
        )

    def set_zero(self, x, x2=None) -> SSpectrum:
        """
        Set x (or range of x) at which y (or y average) is set to 0.

        :param x: value at which y is set to zero
        :param x2: end of range (unless None)
        :return: zeroed SSpectrum
        """
        raise NotImplementedError()

    def sliced(self, start=None, end=None) -> SSpectrum:
        """
        Return a new SSpectrum that is a slice of self.

        :param start: the start of the slice.
        :param end: the end of the slice.
        :return: a new SSpectrum.
        """
        raise NotImplementedError()

    @property
    def norm(self) -> float:
        """
        Determine the Frobenius norm of the SSpectrum.
        """
        raise NotImplementedError()

    def normed(self, target="area", target_value=1) -> SSpectrum:
        """
        Return a normalized SSpectrum.

        :param target:
            'area' - normalize using total area
            'max' - normalize based on max value
            x-value - normalize based on the y-value at this x-value
            (start, end) - normalize based on integration from start to end
        :param target_value: what to normalize the target to
        :return: normalized SSpectrum
        """
        raise NotImplementedError()

    def peaks(
        self,
        indices=False,
        height=None,
        threshold=None,
        distance=None,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=0.5,
        plateau_size=None,
    ) -> list[float]:
        """
        Find the indices of peaks.

        Note: Utilizes scipy.signal.find_peaks and the parameters therein.

        :param indices: return peak indices instead of x-values
        :return: peak x-values (or peak indices if indices == True), properties
        """
        raise NotImplementedError()

    def convert(self, width: float, npoints=10000) -> Spectrum:
        """
        Convert a SSpectrum to a Spectrum
        """
        x_min, x_max = self.domain
        energies = np.linspace(x_min - width * 4, x_max + width * 4, npoints)
        intensities = sum(gaussian(energy, intensity, width, energies) for energy, intensity in self)
        if TYPE_CHECKING:
            assert isinstance(intensities, np.ndarray)

        return Spectrum(self.name, energies, intensities, self.units, self.style, self.time)


def sspectra_from_csvs(*inps: str, names: list[str] = None) -> list[SSpectrum]:
    """
    Read from csvs.

    :param inps: file names of the csvs
    :param names: names of the SSpectra
    :return: list of SSpectra
    """
    ns, energies, intensities = read_csvs(inps)
    names = ns if names is None else names
    return [SSpectrum(name, energies, intensities) for name, xs, ys in zip(names, energies, intensities)]
