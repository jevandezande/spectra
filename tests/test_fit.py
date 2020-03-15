import sys
import numpy as np

from pytest import raises

sys.path.insert(0, '..')

from spectra.fit import *
from spectra.spectrum import spectra_from_csvs


def setup():
    pass


def teardown():
    pass


def test_guess_model():
    spectrum = spectra_from_csvs('tests/files/spectrum1.csv')[0]
    spectrum.style = 'XRD'

    model1, params1 = guess_model(spectrum)
    model2, params2 = XRD_guess_model(spectrum)

    assert len(params1) == 37
    assert params1 == params2


def test_XRD_guess_model():
    spectrum = spectra_from_csvs('tests/files/spectrum1.csv')[0]

    model, params = XRD_guess_model(spectrum)
    assert len(params) == 37


def test_fit_spectrum(tmp_path):
    spectrum = spectra_from_csvs('tests/files/spectrum1.csv')[0]

    with raises(NotImplementedError):
        fit_spectrum(spectrum)

    spectrum.style = 'FTIR'
    with raises(NotImplementedError):
        fit_spectrum(spectrum)

    spectrum.style = 'XRD'
    fit = fit_spectrum(spectrum)
    assert len(fit.params) == 37


def test_plot_fit():
    spectrum = spectra_from_csvs('tests/files/spectrum1.csv')[0]
    spectrum.style = 'XRD'
    fit = fit_spectrum(spectrum)

    fig, ax = plot_fit(fit, spectrum.style)
    plot_fit(fit, spectrum.style, plot=(fig, ax), verbose=True, title='XRD - test_plot_fit')
