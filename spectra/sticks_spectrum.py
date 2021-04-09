from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._abc_spectrum import Spectrum
from .conv_spectrum import ConvSpectrum
from .shapes import gaussian


class SticksSpectrum(Spectrum):
    """
    A SticksSpectrum is a collection of intensities at various energies
    These may be convolved with a shape to produce a ConvSpectrum.
    """

    def __init__(
        self,
        name: str,
        energies: np.ndarray,
        intensities: np.ndarray,
        units: str = None,
        style: str = None,
        time=None,
        y_shift: float = 0,
    ):
        super().__init__(name, energies, intensities, units, style, time)
        self.y_shift = y_shift

    def __rsub__(self, other: float) -> SticksSpectrum:
        """
        !!!Warning, different definition than ConvSpectrum!!!
        """
        copy = self.copy()
        copy.intensities = -copy.intensities + other
        return copy

    def __sub__(self, other: SticksSpectrum | float) -> SticksSpectrum:
        """
        !!!Warning, different definition than ConvSpectrum!!!
        """
        if issubclass(other.__class__, SticksSpectrum):
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
        elif isinstance(other, Spectrum):
            raise NotImplementedError(f"Cannot subtract Spectra of different types: {type(self)=} != {type(other)=}")

        new = self.copy()
        new.y_shift -= other
        return new

    def __add__(self, other: SticksSpectrum | float) -> SticksSpectrum:
        """
        !!!Warning, different definition than ConvSpectrum!!!
        """
        if issubclass(other.__class__, SticksSpectrum):
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
        elif isinstance(other, Spectrum):
            raise NotImplementedError(f"Cannot add Spectra of different types: {type(self)=} != {type(other)=}")

        new = self.copy()
        assert isinstance(new, SticksSpectrum)
        new.y_shift += other
        return new

    def copy(self) -> SticksSpectrum:
        """
        Create a copy of the SticksSpectrum.

        :return: duplicate SticksSpectrum
        """
        copy = super().copy()
        assert isinstance(copy, SticksSpectrum)
        copy.y_shift = self.y_shift
        return copy

    @property
    def domain(self) -> tuple[float, float]:
        """
        Domain of the SticksSpectrum (range of energies).

        :return: min energy, max energy
        """
        return float(self.energies.min()), float(self.energies.max())

    def baseline_subtracted(self, val: float = None) -> SticksSpectrum:
        """
        Return a new SticksSpectrum with the baseline subtracted.

        :param val: amount to subtract, if None, use the lowest value.
        :return: SticksSpectrum with the baseline subtracted.
        """
        new = self.copy()
        new.intensities -= min(self.intensities) if val is None else val
        return new

    @property
    def norm(self) -> float:
        """
        Determine the Frobenius norm of the SticksSpectrum.
        """
        raise NotImplementedError()

    def normed(
        self,
        target: tuple[float, float] | float | str = "max",
        target_value: float = 1,
    ) -> SticksSpectrum:
        if target == "area" or isinstance(target, tuple):
            raise ValueError(f"Could not normalize a SticksSpectrum with {target=}")

        s = super().normed(target, target_value)
        assert isinstance(s, SticksSpectrum)
        return s

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

    def set_zero(self, energy: float, energy2: float = None) -> SticksSpectrum:
        raise NotImplementedError()

    def convert(self, width: float, npoints=10000) -> ConvSpectrum:
        """
        Convert a SticksSpectrum to a ConvSpectrum
        """
        energy_min, energy_max = self.domain
        energies = np.linspace(energy_min - width * 4, energy_max + width * 4, npoints)
        intensities = sum(gaussian(energy, intensity, width, energies) for energy, intensity in self)
        if TYPE_CHECKING:
            assert isinstance(intensities, np.ndarray)

        return ConvSpectrum(self.name, energies, intensities, self.units, self.style, self.time)

    def sliced(self, start: float = None, end: float = None) -> SticksSpectrum:
        new = super().sliced(start, end)
        assert isinstance(new, SticksSpectrum)
        new.y_shift = self.y_shift
        return new

    def smoothed(self, box_pts: int | bool = True) -> SticksSpectrum:
        raise NotImplementedError()