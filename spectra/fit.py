from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from lmfit import Parameters, models
from lmfit.models import Model
from .plot import setup_axis
from .tools import integrate
from .spectrum import Spectrum


def fit_spectrum(
    spectrum: Spectrum,
    style: str = None,
    model: Model = None,
    params: dict = None,
    peak_args: dict = None,
) -> Model:
    """
    Fit a given Spectrum.

    Note: guessing the fit is useful for determining the initial fit, but it is
    always recommended to perform multiple rounds of observing and
    hand-optimizing the fit to ensure it is performed properly.

    :param spectrum: the spectrum to be fit
    :param style: the style of Spectrum (used as a hint for guessing the fit)
    :param model: model to be used
    :param params: params used in the model
    :param peak_args: peak picking arguments to be used for guessing the model
    :return: fitted Model
    """
    if model is None:
        model, params = guess_model(spectrum, style, peak_args)

    return model.fit(spectrum.ys, params, x=spectrum.xs)


def guess_model(
    spectrum: Spectrum, style: str = None, peak_args: dict = None
) -> tuple[Model, dict]:
    """
    Return a guess model of the correct style.

    :param spectrum: the Spectrum to be fit
    :param style: the style of Spectrum (used as a hint for guessing the fit)
    :param peak_args: peak picking arguments to be used for guessing the model
    :return: Model, parameters
    """
    style = spectrum.style if style is None else style

    if style == "XRD":
        return XRD_guess_model(spectrum, peak_args)
    elif style == "IR":
        return IR_guess_model(spectrum, peak_args)

    raise NotImplementedError(f"Does not yet know how to guess a fit for {style}.")


def XRD_guess_model(spectrum: Spectrum, peak_args: dict = None) -> tuple[Model, dict]:
    """
    Guess a fit for the XRD spectrum based on its peaks.

    :param spectrum: the Spectrum to be fit
    :param peak_args: arguments for finding peaks
    :return: Model, parameters
    """
    min_x = spectrum.xs[0]
    max_x = spectrum.xs[-1]
    range_x = max_x - min_x
    max_y = np.max(spectrum.ys)
    min_y = np.min(spectrum.ys)
    range_y = max_y - min_y

    XRD_peak_defaults = {
        "prominence": 0.02 * range_y,
    }
    peak_args = (
        XRD_peak_defaults if peak_args is None else {**XRD_peak_defaults, **peak_args}
    )

    peak_indices, peak_properties = spectrum.peaks(**peak_args, indices=True)

    params = Parameters()
    composite_model = None

    # Fit the peaks
    i = -1
    for i, peak_idx in enumerate(peak_indices):
        prefix = f"c{i}_"
        model = models.VoigtModel(prefix=prefix)
        center = spectrum.xs[peak_idx]
        height = spectrum.ys[peak_idx]

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
    model.set_param_hint("amplitude", min=1, max=max_y)
    model.set_param_hint("center", min=min_x, max=max_x)
    model.set_param_hint("sigma", min=0.1, max=range_x / 2)
    peak_params = {
        f"{prefix}amplitude": max_y / 4,
        f"{prefix}center": 20,
        f"{prefix}sigma": 5,
    }
    params = params.update(model.make_params(**peak_params))
    composite_model += model

    # Add a broader amorphous background peak
    prefix = f"b{i+2}_"
    model = models.ExponentialModel(prefix=prefix)
    model.set_param_hint("amplitude", min=1, max=max_y)
    peak_params = {
        f"{prefix}amplitude": spectrum.ys[:10].mean() * 0.8,
        f"{prefix}decay": 30,
    }
    model.set_param_hint("amplitude", min=1, max=spectrum.ys[:10].mean() * 2)
    model.set_param_hint("decay", min=10, max=100)

    params = params.update(model.make_params(**peak_params))
    composite_model += model

    return composite_model, params


def IR_guess_model(spectrum: Spectrum, peak_args: dict = None) -> tuple[Model, dict]:
    """
    Guess a fit for the IR spectrum based on its peaks.

    :param spectrum: the Spectrum to be fit
    :param peak_args: arguments for finding peaks
    :return: Model, parameters
    """
    max_y = np.max(spectrum.ys)
    min_y = np.min(spectrum.ys)
    range_y = max_y - min_y

    IR_peak_defaults = {
        "prominence": 0.1 * range_y,
    }
    peak_args = (
        IR_peak_defaults if peak_args is None else {**IR_peak_defaults, **peak_args}
    )

    peak_indices, peak_properties = spectrum.peaks(**peak_args, indices=True)

    params = Parameters()
    composite_model = None

    # Fit the peaks
    for i, peak_idx in enumerate(peak_indices):
        prefix = f"a{i}_"
        model = models.GaussianModel(prefix=prefix)
        center = spectrum.xs[peak_idx]
        height = spectrum.ys[peak_idx]

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
    plot: tuple = None,
    verbose: bool = False,
    **setup_axis_args,
) -> tuple:
    """
    Plot the results of fitting a Spectrum.

    :param model: the model to plot
    :param style: the style of Spectrum (used as a hint for guessing the fit)
    :param plot: (figure, axis) on which to plot, generates new figure if None
    :param verbose: print the parameters of the model
    :param setup_axis_args: arguments to be passed to setup_axis
    :return: fig, ax
    """
    if plot is None:
        fig, ax = plt.subplots()
        setup_axis(ax, style, **setup_axis_args)
    else:
        fig, ax = plot

    xs = model.userkws["x"]
    ys = model.data

    ax.scatter(xs, ys, s=1, label="Spectrum")
    ax.plot(xs, model.best_fit, label="Optimized")

    components = model.eval_components()
    if verbose:
        print(f"{'Function':13s}: Initial → Final: Portion of Total, Portion of Fit")
        for name in model.init_values:
            print(
                f"{name:13s}: {model.init_values[name]:7.2f} → {model.best_values[name]:7.2f}"
            )

    if style == "XRD":
        area = {
            "crystalline": 0,
            "amorphous": 0,
            "background": 0,
            "initial": integrate(xs, model.init_fit),
            "optimized": integrate(xs, model.best_fit),
            "total": integrate(xs, ys),
        }
    elif style == "IR":
        area = {
            "absorption": 0,
            "background": 0,
            "optimized": integrate(xs, model.best_fit),
            "total": integrate(xs, ys),
        }
    else:
        raise NotImplementedError(f"Does not yet know how to plot a model for {style}.")

    if verbose:
        print()
        print("Name        | Ratio of model area | Ratio of total area (-background)")
        print("-" * 69)

    for name, vals in components.items():
        name = name[:-1]
        peak_area = integrate(xs, vals)
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
            raise TypeError(
                f'Mismatch component "{name[0]}" and area types. Does the model type match the plot type?'
            )

        ax.plot(xs, vals, linestyle=linestyle, label=name)

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
