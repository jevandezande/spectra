import sys

sys.path.insert(0, '..')

from spectra.plot import *
from spectra.spectrum import Spectrum


def setup():
    pass


def teardown():
    pass


def test_setup_axis():
    fig, ax = plt.subplots()

    setup_axis(ax, 'None', xticks=range(100), xlim=(0, 100))
    setup_axis(ax, 'IR')
    setup_axis(ax, 'UV-vis')
    setup_axis(ax, 'MS')


def test_cycle_values():
    assert next(cycle_values(None)) is None
    assert next(cycle_values(1)) == 1

    it = cycle_values([0, 1, 2])
    assert next(it) == 0
    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 0


def test_plotter():
    s1 = Spectrum('A', np.arange(10), np.arange(10))
    s2 = Spectrum('B', np.arange(10), -np.arange(10))

    assert s1 != s2

    spectra = [s1, s2]

    fig, ax = plt.subplots()
    plotter(spectra,
            title='Hello World', style='IR',
            baseline_subtracted=True, normalized=False, smoothed=False,
            plot=(fig, ax), xlim=(10, 0), xticks=None,
            legend=True, colors=None, markers=None,
            savefig=None
    )
