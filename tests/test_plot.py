import matplotlib.pyplot as plt
import numpy as np
from pytest import raises

from spectra.conv_spectrum import ConvSpectrum
from spectra.plot import cycle_values, plotter, setup_axis


def setup():
    pass


def teardown():
    pass


def test_setup_axis():
    fig, ax = plt.subplots()

    setup_axis(ax, None, xticks=range(100), xlim=(0, 100))
    setup_axis(ax, "ir")
    setup_axis(ax, "RAMAN")
    setup_axis(ax, "Uv-ViS", xticks_minor=True)
    setup_axis(ax, "gC")
    setup_axis(ax, "cHrOmAtOgRaM")
    setup_axis(ax, "mS")
    setup_axis(ax, "NmR")
    setup_axis(ax, "1H-NMR")
    setup_axis(ax, "13C-NMR")
    setup_axis(ax, "XRD")
    with raises(NotImplementedError):
        setup_axis(ax, "None")


def test_cycle_values():
    assert next(cycle_values(None)) is None
    assert next(cycle_values(1)) == 1

    it = cycle_values([0, 1, 2])
    assert next(it) == 0
    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 0


def test_plotter(tmp_path):
    s1 = ConvSpectrum("A", np.arange(10), np.arange(10))
    s2 = ConvSpectrum("B", np.arange(10), np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1]))

    assert s1 != s2

    spectra = [s1, s2]

    fig, ax = plt.subplots()
    plotter(
        spectra,
        title="Hello World",
        style="IR",
        baseline_subtracted=True,
        set_zero=False,
        normalized=False,
        smoothed=False,
        peaks=None,
        plot=(fig, ax),
        xlim=(3500, 800),
        xticks_minor=True,
        yticks_minor=2,
        legend=True,
        colors=None,
        markers=None,
        linestyles=None,
        savefig=f"{tmp_path}/my_IR_figure.png",
    )

    plotter(
        spectra,
        title="World",
        style="UV-Vis",
        baseline_subtracted=True,
        set_zero=False,
        normalized=2,
        smoothed=True,
        peaks=True,
        plot=None,
        xlim=None,
        xticks=None,
        xticks_minor=1,
        yticks_minor=True,
        legend=False,
        colors=["b", "k"],
        markers="x",
        linestyles=["-", ":"],
        savefig=f"{tmp_path}/my_UV-Vis_figure.png",
    )

    plotter(
        spectra,
        title="Hello",
        style="MS",
        baseline_subtracted=False,
        set_zero=7,
        normalized=True,
        smoothed=False,
        peaks=True,
        plot=None,
        xlim=None,
        xticks=None,
        legend=True,
        colors=None,
        markers=None,
        linestyles=None,
        savefig=f"{tmp_path}/my_MS_figure.png",
    )

    plotter(
        spectra,
        title="Hello",
        style="1H-NMR",
        baseline_subtracted=False,
        set_zero=3,
        normalized=False,
        smoothed=True,
        peaks=False,
        plot=None,
        xlim=(0, 10),
        xticks=None,
        legend=False,
        colors=None,
        markers="+",
        linestyles="--",
        savefig=f"{tmp_path}/my_XRD_figure.png",
    )

    xrd_spectra = ConvSpectrum.from_csvs("tests/files/xrd.csv")
    plotter(
        xrd_spectra,
        title="Hello",
        style="XRD",
        baseline_subtracted=False,
        set_zero=7,
        normalized=False,
        smoothed=True,
        peaks=False,
        plot=None,
        xlim=(0, 10),
        xticks=None,
        ylim=(0, 10),
        yticks=(0, 5, 10),
        yticks_minor=True,
        legend=False,
        colors=None,
        markers="+",
        linestyles="--",
        savefig=f"{tmp_path}/my_XRD_figure.png",
    )
