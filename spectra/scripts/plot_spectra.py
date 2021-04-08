#!/usr/bin/env python3
import argparse
import sys
from glob import glob

from matplotlib import pyplot as plt

from spectra.plot import plotter
from spectra.spectrum import spectra_from_csvs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plot the spectra from csv file(s).")
    parser.add_argument("-i", "--input", help="The file(s) to be read (accepts *).", type=str, nargs="+", default=[])
    parser.add_argument("-l", "--limits", help="The limits for the graph, x1, x2.", type=float, nargs="+", default=[])
    parser.add_argument(
        "-p", "--peaks", help="Label the most prominent peaks with their location.", default=False, action="store_true"
    )
    parser.add_argument(
        "-n", "--name", help="The name(s) of the files to be read.", type=str, nargs="+", default="{autogenerate}"
    )
    parser.add_argument("-t", "--spectra_type", help="Type of spectra to plot.", type=str, default="IR")
    parser.add_argument("-s", "--save", help="Where to save the figure.", type=str, default=False)
    parser.add_argument(
        "-b", "--baseline", help="Subtract the baseline.", type=int, nargs="?", const=True, default=False
    )
    parser.add_argument("--smooth", help="Smooth the plots.", type=int, nargs="?", const=True, default=False)
    parser.add_argument(
        "--subtract",
        help="Subtract two Spectra from each other.",
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--title", help="Figure Title", type=str, default=None)
    parser.add_argument(
        "-z",
        "--normalize",
        help="Normalize all plots based on the highest (or selected) peak.",
        type=float,
        nargs="?",
        const=True,
        default=False,
    )
    args = parser.parse_args()

    inps = [i for inp in args.input for i in glob(inp)]
    if not inps:
        print("You must specify file(s) to be read from.")
        sys.exit(1)

    names = list(range(len(inps))) if args.name == "{autogenerate}" else args.name

    spectra = spectra_from_csvs(*inps, names=names)

    assert not (len(args.limits) % 2)
    xlim = args.limits[:2] if args.limits else None
    ylim = args.limits[2:4] if len(args.limits) > 2 else None

    if args.subtract:
        if len(spectra) != 2:
            raise ValueError(f"Can only subtract two spectra from each other, got: {len(spectra)}")

        if args.subtract == "all":
            spectra.append(spectra[0] - spectra[1])
        else:
            spectra = [spectra[0] - spectra[1]]

    fig, ax = plotter(
        spectra,
        title=args.title,
        style=args.spectra_type,
        baseline_subtracted=args.baseline,
        normalized=args.normalize,
        smoothed=args.smooth,
        plot=None,
        xlim=xlim,
        ylim=ylim,
        xticks=None,
        legend=True,
        colors=None,
        markers=None,
        peaks=args.peaks,
        savefig=args.save,
    )

    plt.show()


if __name__ == "__main__":
    main()
