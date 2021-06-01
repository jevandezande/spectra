from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal

from ._abc_spectrum import Spectrum
from .tools import index_of_x, smooth_curve, y_at_x


class ConvSpectrum(Spectrum):
    """
    A ConvSpectrum is a collection of intensities (intensities) at various energies.
    It is a convetional spectrum, but can also be interpretted as a convolved spectrum.
    """

    def __rsub__(self, other: float) -> ConvSpectrum:
        new = self.copy()
        new.name = f"{other} – {self.name}"
        new.intensities[...] = other - self.intensities
        return new

    def __sub__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        intensity_subtractor: np.ndarray | float
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different units.")
            elif self.energies.shape != other.energies.shape:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different shapes.")
            elif any(self.energies != other.energies):
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different energies.")
            other_name = other.name
            intensity_subtractor = other.intensities
        elif isinstance(other, Spectrum):
            raise TypeError(f"Cannot subtract Spectra of different types: {type(self)} != {type(other)=}")
        else:
            other_name = f"{other}"
            intensity_subtractor = other

        new = self.copy()
        new.name = f"{self.name} – {other_name}"
        new.intensities -= intensity_subtractor
        return new

    def __add__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        intensity_adder: np.ndarray | float
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different units.")
            elif self.energies.shape != other.energies.shape:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different shapes.")
            elif any(self.energies != other.energies):
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different energies.")
            other_name = other.name
            intensity_adder = other.intensities
        elif isinstance(other, Spectrum):
            raise TypeError(f"Cannot subtract Spectra of different types: {type(self)} != {type(other)=}")
        else:
            other_name = f"{other}"
            intensity_adder = other

        new = self.copy()
        new.name = f"{self.name} + {other_name}"
        new.intensities += intensity_adder
        return new

    def _intensities(self, energy: float, energy2: Optional[float] = None) -> np.ndarray | float:
        """
        Directly access the intensity-value(s) at energy to energy2.

        :param energy: energy at which to evaluate (or start).
        :param energy2: energy at which to end, if None, only the value at energy is returned.
        :return: intensity or np.ndarray of intensities.
        """
        if energy2 is None:
            return y_at_x(energy, self.energies, self.intensities)
        return self.intensities[index_of_x(energy, self.energies) : index_of_x(energy2, self.energies)]  # type: ignore

    def copy(self) -> ConvSpectrum:
        return super().copy()  # type:ignore

    def smoothed(self, box_pts: int | bool = True) -> ConvSpectrum:
        """
        Make a smoothed version of the ConvSpectrum.

        :param box_pts: number of data points to convolve, if True, use 3
        :return: smoothed ConvSpectrum
        """
        new = self.copy()
        new.intensities[...] = smooth_curve(self.intensities, box_pts)
        return new

    @property
    def range(self) -> tuple[float, float]:
        """
        Determine the range of intensities
        """
        return float(self.intensities.min()), float(self.intensities.max())

    def peaks(
        self,
        indices: bool = False,
        height: Optional[float] = None,
        threshold: Optional[float] = None,
        distance: Optional[float] = None,
        prominence: Optional[float] = None,
        width: Optional[float] = None,
        wlen: Optional[float] = None,
        rel_height: float = 0.5,
        plateau_size: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the indices of peaks.

        Note: Utilizes scipy.signal.find_peaks and the parameters therein.

        :param indices: return peak indices instead of x-values
        :return: peak energies (or peak indices if indices == True), properties
        """
        peaks, properties = signal.find_peaks(
            self.intensities,
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
        return self.energies[peaks], properties
