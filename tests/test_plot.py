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
    setup_axis(ax, 'ir')
    setup_axis(ax, 'Uv-ViS', xticks_minor=True)
    setup_axis(ax, 'gC')
    setup_axis(ax, 'cHrOmAtOgRaM')
    setup_axis(ax, 'mS')
    setup_axis(ax, 'NmR')


def test_cycle_values():
    assert next(cycle_values(None)) is None
    assert next(cycle_values(1)) == 1

    it = cycle_values([0, 1, 2])
    assert next(it) == 0
    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 0


def test_plotter(tmp_path):
    s1 = Spectrum('A', np.arange(10), np.arange(10))
    s2 = Spectrum('B', np.arange(10), -np.arange(10))

    assert s1 != s2

    spectra = [s1, s2]

    fig, ax = plt.subplots()
    plotter(
        spectra,
        title='Hello World', style='IR',
        baseline_subtracted=True, set_zero=False, normalized=False, smoothed=False, peaks=None,
        plot=(fig, ax), xlim=(10, 0), xticks=None,
        legend=True, colors=None, markers=None, linestyles=None,
        savefig=None
    )

    plotter(
        spectra,
        title='Hello World', style='UV-Vis',
        baseline_subtracted=True, set_zero=False, normalized=2, smoothed=True, peaks=True,
        plot=(fig, ax), xlim=None, xticks=(0, 1, 2),
        legend=False, colors=['b', 'k'], markers='x', linestyles=['-', ':'],
        savefig=f'{tmp_path}/my_figure',
    )
