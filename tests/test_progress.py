import sys

sys.path.insert(0, '..')

from glob import glob
from datetime import datetime, timedelta
from spectra.progress import *
from spectra.spectrum import Spectrum, spectra_from_csvs

import pytest


def setup():
    pass


def teardown():
    pass


def test_progress():
    xs1, ys1 = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs1, ys1)

    xs2, ys2 = np.arange(10), np.arange(10)
    s2 = Spectrum('Hello World', xs2, ys2)

    spectra = [s1, s2]

    areas, half_life = progress(spectra, [0, 1])
    assert areas == [1.5, 0.5]
    assert half_life == 1

    areas, half_life = progress(spectra, [3, 7])
    assert areas == [24, 20]
    assert half_life is None


def test_plot_spectra_progress():
    xs1, ys1 = np.arange(10), np.arange(1, 11)
    s1 = Spectrum('Hello World', xs1, ys1)

    xs2, ys2 = np.arange(10), np.arange(10)
    s2 = Spectrum('Hello World', xs2, ys2)

    spectra = [s1, s2]

    fig, axes = plot_spectra_progress(spectra, [1, 2], (3, 4))


@pytest.mark.slow
def test_plot_spectra_progress_slow():
    inputs = glob('tests/files/1-butanol + N 3400/1.00% T12/Round 1/*.CSV')
    strp = lambda x: datetime.strptime(x, '%a %b %d %H-%M-%S %Y')
    timestamps = [strp(inp.split('/')[-1].split(' (')[0]) for inp in inputs]
    # Sort the inputs by the timestamps
    timestamps, inputs = zip(*sorted(zip(timestamps, inputs)))
    times = [(time - timestamps[0]).total_seconds()/(60*60) for time in timestamps]
    spectra = spectra_from_csvs(*inputs)

    timestamps = [strp(inp.split('/')[-1].split(' (')[0]) for inp in inputs]
    fig, axes = plot_spectra_progress(
        spectra, times, (2200, 2500),
        x_units='hours',
        norm = 'max',
    )
