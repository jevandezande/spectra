from typing import Self

import numpy as np

from ._abc_spectrum import NORM_TARGETS, Spectrum
from .continuous_spectrum import ContinuousSpectrum
from .shapes import gaussian


class DiscreteSpectrum(Spectrum):
    """
    A DiscreteSpectrum is a collection of intensities at various energies
    These may be convolved with a shape to produce a ContinuousSpectrum.
    """

    def __sub__(self: Self, other: Spectrum | float) -> Self:
        """
        !!!Warning, different definition than ContinuousSpectrum!!!
        """
        new: Self = self.copy()
        if isinstance(other, DiscreteSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different units.")
            new.name = f"{self.name} – {other.name}"
            new.energies = np.concatenate((self.energies, other.energies))
            new.intensities = np.concatenate((self.intensities, -other.intensities))
        elif isinstance(other, Spectrum):
            raise NotImplementedError(f"Cannot subtract Spectra of different types: {type(self)=} != {type(other)=}")
        else:
            assert isinstance(new, Spectrum)
            new.name = f"{self.name} – {other}"
            new.intensities -= other

        return new

    def __add__(self: Self, other: Spectrum | float) -> Self:
        """
        !!!Warning, different definition than ContinuousSpectrum!!!
        """
        new: Self = self.copy()
        if isinstance(other, DiscreteSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different units.")
            new.name = f"{self.name} + {other.name}"
            new.energies = np.concatenate((self.energies, other.energies))
            new.intensities = np.concatenate((self.intensities, other.intensities))
        elif isinstance(other, Spectrum):
            raise NotImplementedError(f"Cannot add Spectra of different types: {type(self)=} != {type(other)=}")
        else:
            assert isinstance(new, DiscreteSpectrum)
            new.name = f"{self.name} + {other}"
            new.intensities += other

        return new

    @property
    def domain(self) -> tuple[float, float]:
        """
        Domain of the DiscreteSpectrum (range of energies).

        :return: min energy, max energy
        """
        return float(self.energies.min()), float(self.energies.max())

    def baseline_subtracted(self: Self, val: float | None = None) -> Self:
        """
        Return a new DiscreteSpectrum with the baseline subtracted.

        :param val: amount to subtract, if None, use the lowest value.
        :return: DiscreteSpectrum with the baseline subtracted.
        """
        new = self.copy()
        new.intensities -= min(self.intensities) if val is None else val
        return new

    @property
    def norm(self) -> float:
        """
        Determine the Frobenius norm of the DiscreteSpectrum.
        """
        raise NotImplementedError()

    def normed(
        self,
        target: NORM_TARGETS = "max",
        target_value: float = 1,
    ) -> Self:
        if target == "area" or isinstance(target, tuple):
            raise NotImplementedError(f"Could not normalize a DiscreteSpectrum with {target=}")

        return super().normed(target, target_value)

    def set_zero(self: Self, energy: float, energy2: float | None = None) -> Self:
        raise NotImplementedError()

    def convert(
        self,
        width: float,
        npoints: int = 10000,
        energy_lim: tuple[float, float] | None = None,
    ) -> ContinuousSpectrum:
        """
        Convert a DiscreteSpectrum to a ContinuousSpectrum
        """
        domain = energy_lim or (self.domain[0] - width * 4, self.domain[1] + width * 4)
        energies = np.linspace(*domain, npoints)

        intensities = np.sum(intensity * gaussian(energy, width, energies) for energy, intensity in self)  # type:ignore

        return ContinuousSpectrum(self.name, energies, intensities, self.units, self.style, self.time)

    def smoothed(self: Self, box_pts: int | bool = True) -> Self:
        raise NotImplementedError()
