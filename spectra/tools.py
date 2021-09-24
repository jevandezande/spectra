from __future__ import annotations

import csv
import itertools
from glob import glob
from typing import TYPE_CHECKING, Generator, Iterable, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import constants

if TYPE_CHECKING:
    from ._abc_spectrum import Spectrum


def read_csv(inp: str, header: bool = True) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Reads a CSV file.

    :param inp: input file
    :param header: inp contains a header
    :return:
        :titles: titles of the columns
        :xs: x-values (1- or 2-dim np.ndarray)
        :ys: y-values (1- or 2-dim np.ndarray, matches x)
    """
    try:
        with open(inp) as f:
            reader = csv.reader(f)
            titles = next(reader) if header else None

            xs, ys = [], []
            for x, *y in reader:
                xs.append(float(x))
                ys.append([float(y_val) for y_val in y])
    except ValueError as e:
        raise ValueError(f"Error reading value in {inp}.") from e

    xs_array = np.array(xs)
    ys_array = np.array(ys).T

    if titles is None:
        titles = [""] * len(xs)

    return titles, xs_array, ys_array


def read_csvs(inps: Iterable[str] | str, header: bool = True) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Read CSV(s)

    :param inps: input file(s) to read
    :param header: inp contains a header
    :return: titles, xs, ys
    """
    titles: list[str] = []
    if isinstance(inps, str):
        titles, xs, ys = read_csv(inps, header)
        titles = titles[1:]
        xs = np.ones(ys.shape) * xs
    else:
        xs_list, ys_list = [], []
        for inp in inps:
            ts, xs, ys = read_csv(inp, header)
            xs = np.ones(ys.shape) * xs
            titles.extend(ts[1:])
            if ys.shape[1] == 1:
                xs_list.append(xs)
                ys_list.append(ys)
            else:
                for x_vals, y_vals in zip(xs, ys):
                    xs_list.append(x_vals)
                    ys_list.append(y_vals)
        xs = np.array(xs_list)
        ys = np.array(ys_list)

    # Sanity checks
    assert len(xs) == len(ys)
    assert len(ys) == len(titles)

    return titles, xs, ys


def glob_read_csvs(
    inps: Iterable[str] | str, header: bool = True
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """
    Use glob to find CSV(s) and then reads them.

    :param inps: a string or list of strings that can be read by glob
    :param header: inp contains a header
    :return: titles, xs, ys, file_names
    """
    if isinstance(inps, str):
        inps = [inps]
    file_names = list(itertools.chain(*(glob(inp) for inp in inps)))
    titles, xs, ys = read_csvs(file_names)

    return titles, np.array(xs), np.array(ys), file_names


def y_at_x(x_points: Iterable[float] | float, xs: ArrayLike, ys: ArrayLike) -> np.ndarray | float:
    """
    Determine the y-value at a specified x. If in between xs, choose the first
    past it. Assumes xs are ordered.

    :param x_points: x-value(s) for which the y-value is desired
    :param xs: x-values
    :param ys: y-values
    :return: desired y-value
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if len(xs) != len(ys):
        raise ValueError(f"Mismatched lengths: {len(xs)=} and {len(ys)=}")

    return ys[index_of_x(x_points, xs)]


def index_of_x(x_points: Iterable[float] | float, xs: np.ndarray) -> np.ndarray | int:
    """
    Determine the index of value(s) in an ordered list. If in between xs,
    choose the first past it (larger). Assumes xs are ordered.

    :param x_points: value(s) to find
    :param xs: list to search in
    :return: index of the nearest x_point
    """
    # If in reverse order
    revd = xs[0] > xs[-1]
    if revd:
        xs = xs[::-1]

    x_iter = x_points if isinstance(x_points, Iterable) else [x_points]
    for x in x_iter:
        if x < xs[0] or x > xs[-1]:
            raise IndexError(f"x_points not in xs, x_points: {x}, xs: ({xs[0]}→{xs[-1]})")

    return np.searchsorted(xs, x_points) if not revd else len(xs) - np.searchsorted(xs, x_points) - 1  # type: ignore


def integrate(xs: np.ndarray, ys: np.ndarray, x_range: Optional[tuple[float, float]] = None) -> float:
    """
    Integrate a set of ys on the xs.

    Note: if x_range does not fall exactly on values in x, it finds the next largest x value.

    :param xs: x-values
    :param ys: y-values
    :param x_range: range of x_values to integrate over
    :return: integration
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must be of the same length, got: {len(xs)} and {len(ys)}")

    if x_range is not None:
        begin, end = x_range
        if begin < xs[0]:
            raise IndexError(f"x_range starts before first value in xs ({begin} > {xs[0]}")

        start = index_of_x(begin, xs)
        finish = index_of_x(end, xs)

        if TYPE_CHECKING:
            assert isinstance(start, int)
            assert isinstance(finish, int)

        xs = xs[start : finish + 1]
        ys = ys[start : finish + 1]

    return np.trapz(ys, xs)


def smooth_curve(ys: Sequence[float] | np.ndarray, box_pts: int | bool = True) -> np.ndarray:
    """
    Smooth a curve.

    Assumes that the ys are uniformly distributed. Returns output of length
    `max(ys, box_pts)`, boundary effects are visible.

    Note: ys must be > box_pts

    :param ys: points to smooth
    :param box_pts: number of data points to convolve, if True, use 3
    :return: smoothed points
    """
    if box_pts is True:
        box_pts = 3

    box = np.ones(box_pts) / box_pts
    return np.convolve(ys, box, mode="same")


def cull(vals: Sequence, n: int) -> Generator:
    """
    Cull `vals` to have `n` "evenly" spaced values.
    If not evenly divisible, spread them out as evenly as possible.

    :var vals: the values to cull
    :var n: number of values to keep
    :yield: culled values
    """
    yield from (vals[i] for i in np.linspace(0.5, len(vals) - 0.5, n, dtype=int))


def boltzmann_factors(energies: Sequence[float], T: float = 300) -> np.ndarray:
    """
    Compute the Boltzmann factors.

    :param energies: energies in Hartree with which to generate weights
    :param T: temperature, defaults to 300
    """
    if T <= 0:
        raise ZeroDivisionError(f"T must be greater than 0, got: {T=}")

    kBT = constants.k * T / constants.physical_constants["Hartree energy"][0]

    zeroed_energies = np.asarray(energies) - min(energies)
    factors = np.exp(-zeroed_energies / kBT)
    return factors / factors.sum()


def boltzmann_weighted(
    spectra: Sequence[Spectrum], energies: Sequence[float], T: float = 300, rename: bool | str = False
) -> Spectrum:
    """
    Combine spectra via Boltzmann weighting.

    :param spectra: spectra to combine
    :param energies: energies of the spectra
    :param T: temperature for weighting, defaults to room temperature
    :param rename: rename the resulting spectrum
    """
    assert len(spectra) > 0
    assert len(spectra) == len(energies)

    spectrum = sum(s * f for s, f in zip(spectra, boltzmann_factors(energies, T)))
    if TYPE_CHECKING:
        assert isinstance(spectrum, Spectrum)

    if rename:
        spectrum.name = "Boltzmann Spectrum" if isinstance(rename, bool) else rename

    return spectrum
