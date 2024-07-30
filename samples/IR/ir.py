#!/usr/bin/env python3
from spectra import ContinuousSpectrum
from spectra.plot import plotter

spectra = ContinuousSpectrum.from_csvs(
    "samples/IR/data/HDI.csv",
    "samples/IR/data/methanol.csv",
    names=["HDI", "Methanol"],
)

fig, ax = plotter(
    spectra,
    title="IR Spectra",
    style="IR",
    baseline_subtracted=True,
    normalized=False,
    smoothed=False,
    plot=None,
    xlim=None,
    xticks=None,
    legend=True,
    colors=None,
    markers=None,
    peaks=True,
    savefig="ir_py.svg",
)
