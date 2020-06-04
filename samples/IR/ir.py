#!/usr/bin/env python3
import sys

sys.path.insert(0, '../../')

from spectra.plot import plotter
from spectra.spectrum import spectra_from_csvs

spectra = spectra_from_csvs('data/HDI.csv', 'data/methanol.csv', names=['HDI', 'Methanol'])

fig, ax = plotter(
    spectra,
    title='IR Spectra', style='IR',
    baseline_subtracted=True, normalized=False, smoothed=False,
    plot=None, xlim=None, xticks=None,
    legend=True, colors=None, markers=None,
    peaks=True,
    savefig='plots/ir_py.svg',
)
