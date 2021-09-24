from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from lmfit import Parameters, models
from lmfit.models import Model
from scipy.optimize import minimize

from .conv_spectrum import ConvSpectrum
from .plot import subplots
from .tools import integrate


def fit_spectrum(
    spectrum: ConvSpectrum,
    style: Optional[str] = None,
    model: Optional[Model] = None,
    params: Optional[dict] = None,
    peak_args: Optional[dict] = None,
) -> Model:
    """
    Fit a given ConvSpectrum.

    Note: guessing the fit is useful for determining the initial fit, but it is
    always recommended to perform multiple rounds of observing and
    hand-optimizing the fit to ensure it is performed properly.

    :param spectrum: the spectrum to be fit
    :param style: the style of ConvSpectrum (used as a hint for guessing the fit)
    :param model: model to be used
    :param params: params used in the model
    :param peak_args: peak picking arguments to be used for guessing the model
    :return: fitted Model
    """
    if model is None:
        model, params = guess_model(spectrum, style, peak_args)

    return model.fit(spectrum.intensities, params, x=spectrum.energies)


def guess_model(
    spectrum: ConvSpectrum, style: Optional[str] = None, peak_args: Optional[dict] = None
) -> tuple[Model, dict]:
    """
    Return a guess model of the correct style.

    :param spectrum: the ConvSpectrum to be fit
    :param style: the style of ConvSpectrum (used as a hint for guessing the fit)
    :param peak_args: peak picking arguments to be used for guessing the model
    :return: Model, parameters
    """
    style = spectrum.style if style is None else style

    if style == "XRD":
        return XRD_guess_model(spectrum, peak_args)
    elif style == "IR":
        return IR_guess_model(spectrum, peak_args)

    raise NotImplementedError(f"Don't know how to guess a fit for {style=}.")


def XRD_guess_model(spectrum: ConvSpectrum, peak_args: Optional[dict] = None) -> tuple[Model, dict]:
    """
    Guess a fit for the XRD spectrum based on its peaks.

    :param spectrum: the ConvSpectrum to be fit
    :param peak_args: arguments for finding peaks
    :return: Model, parameters
    """
    min_energy, max_energy = spectrum.domain
    range_energies = max_energy - min_energy
    min_intensity, max_intensity = spectrum.range
    range_intensities = max_intensity - min_intensity

    XRD_peak_defaults = {
        "prominence": 0.02 * range_intensities,
    }
    peak_args = XRD_peak_defaults if peak_args is None else {**XRD_peak_defaults, **peak_args}

    peak_indices, peak_properties = spectrum.peaks(**peak_args, indices=True)

    params = Parameters()
    composite_model = None

    # Fit the peaks
    i = -1
    for i, peak_idx in enumerate(peak_indices):
        prefix = f"c{i}_"
        model = models.VoigtModel(prefix=prefix)
        center = spectrum.energies[peak_idx]
        height = spectrum.intensities[peak_idx]

        model.set_param_hint("amplitude", min=100, max=1.1 * height)
        model.set_param_hint("center", min=center - 1, max=center + 1)
        model.set_param_hint("sigma", min=0.01, max=0.3)
        peak_params = {
            f"{prefix}amplitude": height * 0.8,
            f"{prefix}center": center,
            f"{prefix}sigma": 0.1,
        }

        params = params.update(model.make_params(**peak_params))
        composite_model = model if composite_model is None else composite_model + model

    # Add a broad amorphous peak
    prefix = f"a{i+1}_"
    model = models.VoigtModel(prefix=prefix)
    model.set_param_hint("center", min=min_energy, max=max_energy)
    model.set_param_hint("amplitude", min=1, max=max_intensity)
    model.set_param_hint("sigma", min=0.1, max=range_energies / 2)
    peak_params = {
        f"{prefix}amplitude": max_intensity / 4,
        f"{prefix}center": 20,
        f"{prefix}sigma": 5,
    }
    params = params.update(model.make_params(**peak_params))
    composite_model += model

    # Add a broader amorphous background peak
    prefix = f"b{i+2}_"
    model = models.ExponentialModel(prefix=prefix)
    model.set_param_hint("amplitude", min=1, max=max_intensity)
    peak_params = {
        f"{prefix}amplitude": spectrum.intensities[:10].mean() * 0.8,
        f"{prefix}decay": 30,
    }
    model.set_param_hint("amplitude", min=1, max=spectrum.intensities[:10].mean() * 2)
    model.set_param_hint("decay", min=10, max=100)

    params = params.update(model.make_params(**peak_params))
    composite_model += model

    return composite_model, params


def IR_guess_model(spectrum: ConvSpectrum, peak_args: Optional[dict] = None) -> tuple[Model, dict]:
    """
    Guess a fit for the IR spectrum based on its peaks.

    :param spectrum: the ConvSpectrum to be fit
    :param peak_args: arguments for finding peaks
    :return: Model, parameters
    """
    min_intensity, max_intensity = spectrum.range
    range_intensities = max_intensity - min_intensity

    IR_peak_defaults = {
        "prominence": 0.1 * range_intensities,
    }
    peak_args = IR_peak_defaults if peak_args is None else {**IR_peak_defaults, **peak_args}

    peak_indices, peak_properties = spectrum.peaks(**peak_args, indices=True)

    params = Parameters()
    composite_model = None

    # Fit the peaks
    for i, peak_idx in enumerate(peak_indices):
        prefix = f"a{i}_"
        model = models.GaussianModel(prefix=prefix)
        center = spectrum.energies[peak_idx]
        height = spectrum.intensities[peak_idx]

        model.set_param_hint("amplitude", min=0.05 * height)
        model.set_param_hint("center", min=center - 10, max=center + 10)
        model.set_param_hint("sigma", min=0.1, max=100)
        peak_params = {
            f"{prefix}amplitude": height * 0.8,
            f"{prefix}center": center,
            f"{prefix}sigma": 10,
        }

        params = params.update(model.make_params(**peak_params))
        composite_model = model if composite_model is None else composite_model + model

    return composite_model, params


def plot_fit(
    model: Model,
    style: str,
    plot: Optional[tuple] = None,
    verbose: bool = False,
    **setup_axis_kw,
) -> tuple:
    """
    Plot the results of fitting a ConvSpectrum.

    :param model: the model to plot
    :param style: the style of ConvSpectrum (used as a hint for guessing the fit)
    :param plot: (figure, axis) on which to plot, generates new figure if None
    :param verbose: print the parameters of the model
    :param setup_axis_args: arguments to be passed to setup_axis
    :return: fig, ax
    """
    if plot is None:
        fig, ((ax,),) = subplots(style, setup_axis_kw=setup_axis_kw)
    else:
        fig, ax = plot

    energies = model.userkws["x"]
    intensities = model.data

    ax.scatter(energies, intensities, s=1, label="Spectrum")
    ax.plot(energies, model.best_fit, label="Optimized")

    components = model.eval_components()
    if verbose:
        print(f"{'Function':13s}: Initial → Final: Portion of Total, Portion of Fit")
        for name in model.init_values:
            print(f"{name:13s}: {model.init_values[name]:7.2f} → {model.best_values[name]:7.2f}")

    if style == "XRD":
        area = {
            "crystalline": 0,
            "amorphous": 0,
            "background": 0,
            "initial": integrate(energies, model.init_fit),
            "optimized": integrate(energies, model.best_fit),
            "total": integrate(energies, intensities),
        }
    elif style == "IR":
        area = {
            "absorption": 0,
            "background": 0,
            "optimized": integrate(energies, model.best_fit),
            "total": integrate(energies, intensities),
        }
    else:
        raise NotImplementedError(f"Does not yet know how to plot a model for {style}.")

    if verbose:
        print()
        print("Name        | Ratio of model area | Ratio of total area (-background)")
        print("-" * 69)

    for name, vals in components.items():
        name = name[:-1]
        peak_area = integrate(energies, vals)
        try:
            if name[0] == "c":
                area["crystalline"] += peak_area
                linestyle = "-"
            elif name[0] == "a":
                if style == "XRD":
                    area["amorphous"] += peak_area
                    linestyle = "--"
                elif style == "IR":
                    area["absorption"] += peak_area
                    linestyle = "-"
            elif name[0] == "b":
                area["background"] += peak_area
                linestyle = "--"
            else:
                raise ValueError(f"Not sure what to do with peak named: {name}.")
        except KeyError:
            raise TypeError(f'Mismatch component "{name[0]}" and area types. Does the model type match the plot type?')

        ax.plot(energies, vals, linestyle=linestyle, label=name)

        if verbose:
            print(
                f"{name:11s} | {peak_area/area['optimized']:>13.3f}       |"
                + " {peak_area/(area['total'] - area['background']):>11.3f}"
            )

    if verbose:
        print("-" * 69)
        for name, n_area in area.items():
            print(
                f"{name:11s} | {n_area/area['optimized']:>13.3f}       |"
                + " {n_area/(area['total'] - area['background']):>11.3f}"
            )
        print()

    fig.legend()

    return fig, ax


def fit_with_spectra(
    target: ConvSpectrum, *spectra: ConvSpectrum, x0: Optional[Iterable] = None, **kwargs
) -> np.ndarray:
    for s in spectra:
        assert all(s.energies == spectra[0].energies)
    assert len(target) == len(spectra[0])
    if x0 is None:
        x0 = np.ones(len(spectra))
    else:
        x0 = np.asarray(x0)
        assert len(x0) == len(spectra)

    if "bounds" not in kwargs:
        kwargs["bounds"] = [(0, None)] * len(x0)

    intensities = np.array([s.intensities for s in spectra])

    def func(weights):
        return np.linalg.norm(target.intensities - sum(w * intensity for w, intensity in zip(weights, intensities)))

    return minimize(func, x0, **kwargs)
