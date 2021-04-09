from __future__ import annotations

import numpy as np
from scipy import signal

from ._abc_spectrum import Spectrum
from .tools import index_of_x, smooth_curve, y_at_x


class ConvSpectrum(Spectrum):
    """
    A ConvSpectrum is a collection of intensities (intensities) at various energies.
    It is a convetional spectrum, but can also be interpretted as a convolved spectrum.
    """

    def __rsub__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        return self.__class__(f"{self.name}", np.copy(self.energies), other - self.intensities)  # type: ignore

    def __sub__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different units.")
            elif self.energies.shape != other.energies.shape:
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different shapes.")
            elif any(self.energies != other.energies):
                raise NotImplementedError(f"Cannot subtract {self.__class__.__name__} with different energies.")
            return self.__class__(
                f"{self.name} – {other.name}",
                np.copy(self.energies),
                self.intensities - other.intensities,
                units=self.units,
                style=self.style,
            )
        return self.__class__(
            f"{self.name}",
            np.copy(self.energies),
            self.intensities - other,
            units=self.units,
            style=self.style,
        )

    def __add__(self, other: ConvSpectrum | float) -> ConvSpectrum:
        if isinstance(other, ConvSpectrum):
            if self.units != other.units:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different units.")
            elif self.energies.shape != other.energies.shape:
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different shapes.")
            elif any(self.energies != other.energies):
                raise NotImplementedError(f"Cannot add {self.__class__.__name__} with different energies.")
            return self.__class__(
                f"{self.name} + {other.name}", np.copy(self.energies), self.intensities + other.intensities
            )
        return self.__class__(
            f"{self.name}",
            np.copy(self.energies),
            self.intensities + other,
            units=self.units,
            style=self.style,
        )

    def _intensities(self, energy: float, energy2: float = None) -> np.ndarray | float:
        """
        Directly access the intensity-value(s) at energy to energy2.

        :param energy: energy at which to evaluate (or start).
        :param energy2: energy at which to end, if None, only the value at energy is returned.
        :return: intensity or np.ndarray of intensities.
        """
        if energy2 is None:
            return y_at_x(energy, self.energies, self.intensities)
        return self.intensities[index_of_x(energy, self.energies) : index_of_x(energy2, self.energies)]  # type: ignore

    def smoothed(self, box_pts: int | bool = True) -> ConvSpectrum:
        """
        Make a smoothed version of the ConvSpectrum.

        :param box_pts: number of data points to convolve, if True, use 3
        :return: smoothed ConvSpectrum
        """
        return self.__class__(
            f"{self.name}",
            np.copy(self.energies),
            smooth_curve(self.intensities, box_pts),
            units=self.units,
            style=self.style,
        )

    @property
    def range(self) -> tuple[float, float]:
        """
        Determine the range of intensities
        """
        return float(self.intensities.min()), float(self.intensities.max())

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
