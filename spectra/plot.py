from itertools import cycle
from typing import Any, Iterable, Iterator, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator
from numpy.typing import ArrayLike

from ._abc_spectrum import Spectrum
from ._typing import OPT_ITER_FLOAT, OPT_ITER_STR, OPT_PLOT, PLOT
from .continuous_spectrum import ContinuousSpectrum
from .discrete_spectrum import DiscreteSpectrum
from .tools import y_at_x


def plotter(
    spectra: Sequence[Spectrum],
    title: str | None = None,
    style: str | None = None,
    baseline_subtracted: float | bool = False,
    set_zero: Any = False,
    normalized: float | bool = False,
    smoothed: bool | int = False,
    peaks: dict | bool = False,
    plot: OPT_PLOT = None,
    xlim: tuple[float, float] | None = None,
    xticks: ArrayLike | None = None,
    xticks_minor: int | bool = True,
    xlabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    yticks: ArrayLike | None = None,
    yticks_minor: int | bool = True,
    ylabel: str | None = None,
    labels: OPT_ITER_STR = None,
    colors: OPT_ITER_STR = None,
    alphas: OPT_ITER_FLOAT = None,
    markers: OPT_ITER_STR = None,
    linestyles: OPT_ITER_STR = None,
    linewidths: OPT_ITER_FLOAT = None,
    legend: bool = True,
    savefig: str | None = None,
) -> PLOT:
    """
    Plot a list of Spectra.

    :param spectra: Spectra to plot
    :param title: title of the plot
    :param style: plot-style (e.g. IR, UV-Vis)
    :param baseline_subtracted: amount to subtract, if True, use the lowest value from each spectra
    :param set_zero: set x_value (or range of values) at which y (or y average) is set to 0.
    :param normalized: normalize all of the curves at given point (or highest if True)
    :param smoothed: number of points with which to smooth
    :param peaks: dictionary of peak picking parameters
    :param plot: where to plot, generates new figure if None
    :param x*: x-axis setup parameters
    :param y*: y-axis setup parameters
    :param labels: labels for the spectra, if None, generates based on the spectrum name
    :param colors: colors to plot the spectra
    :param alphas: transparency settings to use
    :param markers: markers to plot the spectra
    :param linestyles: linestyles to plot the spectra
    :param linewidths: linewidths to plot the spectra
    :param legend: whether to plot a legend
    :param savefig: where to save the figure
    :return: figure and axes
    """
    assert all(isinstance(s, type(spectra[0])) for s in spectra[1:])

    assert not (baseline_subtracted and set_zero)

    if style is None:
        style = spectra[0].style
        assert style

    if baseline_subtracted:
        assert all(isinstance(s, ContinuousSpectrum) for s in spectra)
        if baseline_subtracted is True:
            spectra = [s.baseline_subtracted(baseline_subtracted) for s in spectra]
    elif set_zero:
        assert all(isinstance(s, ContinuousSpectrum) for s in spectra)
        x, x2 = set_zero if isinstance(set_zero, Iterable) else (set_zero, None)
        spectra = [s.set_zero(x, x2) for s in spectra]

    if normalized:
        assert all(isinstance(s, ContinuousSpectrum) for s in spectra)
        if normalized is True:
            spectra = [s / max(s.intensities) for s in spectra]
        else:
            spectra = [s / y_at_x(normalized, s.energies, s.intensities) for s in spectra]

    if smoothed:
        assert all(isinstance(s, ContinuousSpectrum) for s in spectra)
        spectra = [s.smoothed(smoothed) for s in spectra]

    if plot is None:
        fig, ((ax,),) = subplots(style)  # type: ignore
    else:
        fig, ax = plot

    setup_axis(
        ax,
        style,
        title,
        xlim,
        xticks,
        xticks_minor,
        xlabel,
        ylim,
        yticks,
        yticks_minor,
        ylabel,
    )

    plot_spectra(
        spectra,
        style,
        ax,
        labels=labels,
        colors=colors,
        alphas=alphas,
        markers=markers,
        linestyles=linestyles,
        linewidths=linewidths,
        peaks=peaks,
    )

    if legend:
        ax.legend()

    if savefig:
        fig.savefig(savefig)

    return fig, ax


def plot_spectra(
    spectra: Sequence[Spectrum],
    style: str,
    ax: Axes,
    labels: OPT_ITER_STR = None,
    colors: OPT_ITER_STR = None,
    alphas: OPT_ITER_FLOAT = None,
    markers: OPT_ITER_STR = None,
    linestyles: OPT_ITER_STR = None,
    linewidths: OPT_ITER_FLOAT = None,
    peaks: dict | bool = False,
):
    """
    Plot Spectra on an axis.

    :param spectra: the Spectra to be plotted
    :param ax: the axis on which to plot
    :param style: the plot style
    :param labels: labels for the spectra, if None, generates based on the spectrum name
    :param colors: the colors to use
    :param alphas: transparency settings to use
    :param markers: the markers to use at each point on the plot
    :param linestyles: the styles of line to use
    :param linewidths: the widths of line to use
    :param peaks: peak highlighting parameters
    """
    properties = (labels, colors, alphas, markers, linestyles, linewidths)
    for spectrum, label, color, alpha, marker, linestyle, linewidth in zip(spectra, *map(cycle_values, properties)):
        plot_spectrum(
            spectrum,
            style,
            ax,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            peaks=peaks,
        )


def plot_spectrum(
    spectrum: Spectrum,
    style: str,
    ax: Axes,
    label: str | None = None,
    color: str | None = None,
    marker: str | None = None,
    linestyle: str | None = None,
    linewidth: float | None = None,
    alpha: float | None = None,
    peaks: dict | bool = False,
):
    """
    Plot a Spectrum on an axis.

    :param spectrum: the Spectrum to be plotted
    :param style: the plot style; if None, generates based on the spectrum style
    :param ax: the axis on which to plot
    :param label: label for the spectrum; if None, generates based on the spectrum name
    :param color: the color to use
    :param marker: the marker to use at each point on the plot
    :param linestyle: the style of line to use
    :param linewidth: the width of line to use
    :param alpha: transparency setting
    :param peaks: peak highlighting parameters
    """
    style = spectrum.style if style is None else style
    label = spectrum.name if label is None else label
    assert style is not None
    assert label is not None

    def bar(*args, **kwargs):
        # delete offending kwargs
        if "marker" in kwargs:
            del kwargs["marker"]
        ax.bar(*args, **kwargs)

    def line(*args, **kwargs):
        ax.plot(*args, **kwargs)

    plot_func = bar if style in ["MS"] or isinstance(spectrum, DiscreteSpectrum) else line

    plot_func(
        spectrum.energies,
        spectrum.intensities,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )

    if peaks:
        assert isinstance(spectrum, ContinuousSpectrum)
        plot_peaks(spectrum, style, ax, color, marker, linestyle, linewidth, peaks)


def plot_peaks(
    spectrum: ContinuousSpectrum,
    style: str,
    ax: Axes,
    color: str | None = None,
    marker: str | None = None,
    linestyle: str | None = None,
    linewidth: float | None = None,
    peaks: dict | Literal[True] = True,
):
    """
    Mark the peaks on the spectrum.

    :param spectrum: the Spectrum to find peaks on
    :param ax: the axis on which to plot
    :param style: the plot style
    :param color: the color to use
    :param marker: the marker to use at each point on the plot
    :param linestyle: the style of line to use
    :param linewidth: the width of line to use
    :param peaks: peak highlighting parameters
    """
    peak_defaults = {
        "format": "4.1f",
        "labels": True,
        "marks": "x",
        "print": True,
        "prominence": -0.1 * np.subtract(*spectrum.range),
    }
    if style == "MS":
        peak_defaults |= {
            "format": "4.0f",
            "marks": None,
            "prominence": -0.05 * np.subtract(*spectrum.range),
        }

    peaks = peak_defaults if peaks is True else peak_defaults | peaks

    peak_indices, _ = spectrum.peaks(True, prominence=peaks["prominence"])
    peak_energies, peak_intensities = (
        spectrum.energies[peak_indices],
        spectrum.intensities[peak_indices],
    )

    if peaks["marks"]:
        ax.scatter(peak_energies, peak_intensities, color=color, marker=peaks["marks"])

    if peaks["labels"]:
        for energy, intensity in zip(peak_energies, peak_intensities):
            ax.text(
                energy,
                intensity,
                f'{{:{peaks["format"]}}}'.format(energy),
                verticalalignment="bottom",
            )

    if peaks["print"]:
        print("  Energies  Intensities")
        for energy, intensity in zip(peak_energies, peak_intensities):
            print(f"{energy:>9.3f}  {intensity:>9.3f}")


def subplots(style: str, *args, setup_axis_kw: dict | None = None, **kwargs) -> PLOT:
    """Make a (non-squeezed) subplots."""
    kwargs["squeeze"] = False

    if "sharex" not in kwargs:
        kwargs["sharex"] = True
    if "sharey" not in kwargs:
        kwargs["sharey"] = True

    gridspec_defaults = {
        "hspace": 0,
        "wspace": 0,
    }
    gridspec_kw = kwargs["gridspec_kw"] if "gridspec_kw" in kwargs else {}
    kwargs["gridspec_kw"] = gridspec_defaults | gridspec_kw

    fig, axes = plt.subplots(*args, **kwargs)

    setup_axis_kw = setup_axis_kw if setup_axis_kw else {}
    setup_axis(axes, style, **setup_axis_kw)

    for i, sub_ax in enumerate(axes):
        for j, ax in enumerate(sub_ax):
            if i != len(axes) - 1:
                ax.set_xlabel(None)
            if j:
                ax.set_ylabel(None)

    return fig, axes


def setup_axis(
    ax: Iterable | Axes,
    style: str | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    xticks: ArrayLike | None = None,
    xticks_minor: int | bool = True,
    xlabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    yticks: ArrayLike | None = None,
    yticks_minor: int | bool = True,
    ylabel: str | None = None,
):
    """
    Set up the axis labels and limits.

    Autogenerates based on style for any variable set to None.

    :param ax: axis to setup
    :param style: style to use
    :param title: title of the axis
    :param *lim: limits for *-axis values
    :param *ticks: *-axis ticks
    :param *ticks_minor: *-axis minor ticks
    :param *label: label for the *-axis
    """
    if not isinstance(ax, Axes):
        for sub_ax in ax:
            setup_axis(
                sub_ax,
                style,
                title,
                xlim,
                xticks,
                xticks_minor,
                xlabel,
                ylim,
                yticks,
                yticks_minor,
                ylabel,
            )

    else:

        def make_ticks(start: float, end: float, tw: float) -> np.ndarray:
            return np.arange(int(start / tw) * tw, int(end / tw + 1) * tw, tw)

        backwards = False
        if style:
            style = style.upper()

            if style == "IR":
                backwards = True
                xlim = xlim or (3500.0, 650.0)
                xticks = xticks or make_ticks(*xlim, -500)
                xlabel = xlabel or "Energy (cm$^{-1}$)"
                ylabel = ylabel or "Absorbance"

            elif style == "RAMAN":
                xlim = xlim or (200.0, 3500.0)
                assert xlim
                xticks = xticks or make_ticks(*xlim, 500)
                xlabel = xlabel or "Energy (cm$^{-1}$)"
                ylabel = ylabel or "Intensity"

            elif style == "UV-VIS":
                xlim = xlim or (200.0, 900.0)
                xticks = xticks or make_ticks(*xlim, 100)
                xlabel = xlabel or "Wavelength (nm)"
                ylabel = ylabel or "Absorbance"

            elif style in ["GC", "HPLC", "CHROMATOGRAM"]:
                xlabel = xlabel or "Time (min)"
                ylabel = ylabel or "Response"

            elif style == "MS":
                xlabel = xlabel or "m/z"
                ylabel = ylabel or "Count"

            elif "NMR" in style:
                backwards = True
                if style == "1H-NMR":
                    xlim = xlim or (10.0, 0.0)
                    xticks = xticks or make_ticks(*xlim, -1)
                elif style == "13C-NMR":
                    xlim = xlim or (200.0, 0.0)
                    xticks = xticks or make_ticks(*xlim, -10)
                xlabel = xlabel or "ppm"

            elif style == "XRD":
                xlim = xlim or (0.0, 50.0)
                xticks = xticks or make_ticks(*xlim, 10)
                xlabel = xlabel or "Diffraction Angle (2θ°)"
                ylabel = ylabel or "Intensity"

            elif style == "XPS":
                backwards = True
                xlim = xlim or (1000.0, 0.0)
                xticks = xticks or make_ticks(*xlim, -100)
                xlabel = xlabel or "Energy (eV)"
                ylabel = ylabel or "Counts"

            else:
                raise NotImplementedError(f"{style=} is not yet implemented, buy a developer a coffee.")

        if title is not None:
            ax.set_title(title)

        if xticks is not None:
            ax.set_xticks(xticks)
        if xticks_minor is True:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        elif xticks_minor is not None:
            xticks_minor = xticks_minor if not backwards else -xticks_minor
            ax.xaxis.set_minor_locator(AutoMinorLocator(xticks_minor))
        if yticks is not None:
            ax.set_yticks(yticks)
        if yticks_minor is True:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        elif yticks_minor is not None:
            ax.yaxis.set_minor_locator(AutoMinorLocator(yticks_minor))

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)


def cycle_values(values: Iterable[Any] | Any) -> Iterator[Any]:
    """
    Make a cycle iterator of values.

    :param values: a value or list of values to be cycled.
    :return: iterator of cycled values
    """
    if not isinstance(values, Iterable):
        values = [values]
    yield from cycle(values)
