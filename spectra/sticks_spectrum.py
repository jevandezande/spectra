from __future__ import annotations

from typing import Optional

import numpy as np

from ._abc_spectrum import Self, Spectrum
from .conv_spectrum import ConvSpectrum
from .shapes import gaussian


class SticksSpectrum(Spectrum):
    """
    A SticksSpectrum is a collection of intensities at various energies
    These may be convolved with a shape to produce a ConvSpectrum.
    """

    def __sub__(self, other: SticksSpectrum | float) -> Self:
        """
        !!!Warning, different definition than ConvSpectrum!!!
        """
        new: Self = self.copy()
        if isinstance(other, SticksSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different units.")
            new.name = f"{self.name} – {other.name}"
            new.energies = np.concatenate((self.energies, other.energies))
            new.intensities = np.concatenate((self.intensities, -other.intensities))
        elif isinstance(other, Spectrum):
            raise NotImplementedError(f"Cannot subtract Spectra of different types: {type(self)=} != {type(other)=}")
        else:
            assert isinstance(new, SticksSpectrum)
            new.name = f"{self.name} – {other}"
            new.intensities -= other

        return new

    def __add__(self, other: SticksSpectrum | float) -> Self:
        """
        !!!Warning, different definition than ConvSpectrum!!!
        """
        new: Self = self.copy()
        if isinstance(other, SticksSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different units.")
            new.name = f"{self.name} + {other.name}"
            new.energies = np.concatenate((self.energies, other.energies))
            new.intensities = np.concatenate((self.intensities, other.intensities))
        elif isinstance(other, Spectrum):
            raise NotImplementedError(f"Cannot add Spectra of different types: {type(self)=} != {type(other)=}")
        else:
            assert isinstance(new, SticksSpectrum)
            new.name = f"{self.name} + {other}"
            new.intensities += other

        return new

    @property
    def domain(self) -> tuple[float, float]:
        """
        Domain of the SticksSpectrum (range of energies).

        :return: min energy, max energy
        """
        return float(self.energies.min()), float(self.energies.max())

    def baseline_subtracted(self, val: Optional[float] = None) -> SticksSpectrum:
        """
        Return a new SticksSpectrum with the baseline subtracted.

        :param val: amount to subtract, if None, use the lowest value.
        :return: SticksSpectrum with the baseline subtracted.
        """
        new: SticksSpectrum = self.copy()
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

        return super().normed(target, target_value)

    def set_zero(self, energy: float, energy2: Optional[float] = None) -> Self:
        raise NotImplementedError()

    def convert(
        self, width: float, npoints: int = 10000, energy_lim: Optional[tuple[float, float]] = None
    ) -> ConvSpectrum:
        """
        Convert a SticksSpectrum to a ConvSpectrum
        """
        domain = energy_lim or (self.domain[0] - width * 4, self.domain[1] + width * 4)
        energies = np.linspace(*domain, npoints)

        intensities = np.sum(intensity * gaussian(energy, width, energies) for energy, intensity in self)  # type:ignore

        return ConvSpectrum(self.name, energies, intensities, self.units, self.style, self.time)

    def smoothed(self, box_pts: int | bool = True) -> Self:
        raise NotImplementedError()
