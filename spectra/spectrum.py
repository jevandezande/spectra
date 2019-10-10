import numpy as np

from .tools import read_csvs, smooth_curve, y_at_x, index_of_x
from scipy import signal


class Spectrum:
    def __init__(self, name, xs, ys, units=''):
        """
        A Spectrum

        :param name: name of the spectrum
        :param xs: x-values
        :param ys: intensities
        :param units: units for the x-values
        """
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)
        assert len(xs.shape) == 1
        assert xs.shape == ys.shape

        self.name = name
        self.xs = xs
        self.ys = ys
        self.units = units

    def __iter__(self):
        """
        Iterate over points in the Spectrum.
        :yield: x, y
        """
        yield from zip(self.xs, self.ys)

    def __eq__(self, other):
        return self.name == other.name \
            and all(self.xs == other.xs) \
            and all(self.ys == other.ys) \
            and self.units == other.units

    def __len__(self):
        """
        Number of points in the Spectrum.
        """
        return len(self.xs)

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.name}>'

    def __str__(self):
        return repr(self)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, Spectrum):
            if type(self) != type(other):
                raise TypeError(f'Cannot add spectra of different types: {type(self)} != {type(other)}.')
            if self.xs != other.xs:
                raise NotImplementedError(f'Cannot add spectra with different x-values')
            return self.__class__(f'{self.name} + {other.name}', np.copy(self.xs), self.ys + other.ys)
        return self.__class__(f'{self.name}', np.copy(self.xs), self.ys + other)

    def __rtruediv__(self, other):
        if isinstance(other, Spectrum):
            return 1 / self.xs * other
        return Spectrum(self.name, self.xs, 1 / self.ys) * other

    def __truediv__(self, other):
        if isinstance(other, Spectrum):
            return 1 / other.ys * self
        return 1 / other * self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, Spectrum):
            if type(self) != type(other):
                raise TypeError(f'Cannot multiply spectra of different types: {type(self)} != {type(other)}.')
            if self.xs != other.xs:
                raise NotImplementedError(f'Cannot multiply spectra with different x-values')
            return self.__class__(f'{self.name} * {other.name}', np.copy(self.xs), self.ys * other.ys)
        return self.__class__(f'{self.name}', np.copy(self.xs), self.ys * other)

    @property
    def max_absorbance(self):
        max_idx = np.argmax(self.ys)
        return self.xs[max_idx], self.ys[max_idx]

    @property
    def domain(self):
        """
        Domain of the Spectrum (range of x-values)

        :return: first x, last x
        """
        return self.xs[0], self.xs[-1]

    def smoothed(self, box_pts=True):
        """
        Generate a smoothed version of the Spectrum.

        :param box_pts: number of data points to convolve, if True, use 3
        :return: smoothed Spectrum
        """
        return self.__class__(self.name, np.copy(self.xs), smooth_curve(self.ys, box_pts))

    def baseline_subtracted(self, val=None):
        """
        Subtract the baseline.

        :param val: amount to subtract, if none, use the lowest value
        :return: Spectrum with the baseline subtracted.
        """
        if val is None:
            val = self.ys.min()
        return self.__class__(self.name, np.copy(self.xs), self.ys - val)

    def set_zero(self, x_val, x2_val=None):
        """
        Set x_value (or range of values) at which y (or y average) is set to 0.

        :param x_val: value at which y is set to zero
        :param x2_val: end of range (unless None)
        :return: zeroed Spectrum
        """
        xs = self.xs
        if x2_val is None:
            y = y_at_x(x_val, self.xs, self.ys)
        else:
            y = np.mean(self.ys[index_of_x(x_val, xs):index_of_x(x2_val, xs)])

        return self.baseline_subtracted(y)

    def sliced(self, start=None, end=None):
        """
        Return a new Spectrum that is a slice of self.

        :param start: the start of the slice.
        :param end: the end of the slice.
        :return: a new Spectrum.
        """
        xs, ys = self.xs, self.ys

        start_i = index_of_x(start, xs) if start is not None else None
        end_i = index_of_x(end, xs) if end is not None else None

        return Spectrum(self.name, xs[start_i:end_i], ys[start_i:end_i], self.units)

    def peaks(self, indices=False, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None,
              rel_height=0.5, plateau_size=None):
        """
        Find the indeces of peaks.

        Utilizes scipy.signal.find_peaks and the parameters therein.

        :param indices: return peak indices instead of x-values
        :return: peak x-values (or peak indices if indices == True), properties
        """
        peaks, properties = signal.find_peaks(self.ys, height, threshold, distance, prominence, width, wlen, rel_height,
                                              plateau_size)
        if indices:
            return peaks, properties
        return self.xs[peaks], properties


def spectra_from_csvs(*inps, names=None):
    """
    Read from a csv. Must only contain two columns: xs and ys.

    :param inps: file names of the csvs
    :param names: names of the Spectra
    :return: list of Spectra
    """
    ns, x_vals, y_vals = read_csvs(inps)
    if names is None:
        names = ns
    return [Spectrum(name, xs, ys) for name, xs, ys in zip(names, x_vals, y_vals)]
