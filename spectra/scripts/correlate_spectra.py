#!/usr/bin/env python3
import argparse
import sys
from glob import glob

from spectra import ContinuousSpectrum


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the spectra from csv file(s).")
    parser.add_argument(
        "-i",
        "--input",
        help="The file(s) to be read (accepts *).",
        type=str,
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "-n",
        "--name",
        help="The name(s) of the files to be read.",
        type=str,
        nargs="+",
        default="{autogenerate}",
    )
    parser.add_argument(
        "--title",
        help="Figure Title",
        type=str,
    )
    args = parser.parse_args()

    inps = [i for inp in args.input for i in glob(inp)]
    if not inps:
        print("You must specify file(s) to be read from.")
        sys.exit(1)

    names: list[str] = list(map(str, range(len(inps)))) if args.name == "{autogenerate}" else args.name

    spectra = ContinuousSpectrum.from_csvs(*inps, names=names)

    if len(spectra) < 2:
        raise ValueError("Need at least two Spectra to correlate.")
    elif len(spectra) == 2:
        print(spectra[0].correlation(spectra[1]))
    else:
        print("  | " + "  | ".join(f"{i:>3d}" for i in range(len(spectra))))
        print("-" * (len(spectra) * 7 + 3))
        for i, s1 in enumerate(spectra):
            line = [f"{i:>2d}"]
            for j, s2 in enumerate(spectra):
                if i < j:
                    line.append("   -  ")
                elif i == j:
                    line.append("   1  ")
                else:
                    line.append(f"{s1.correlation(s2):>6.3f}")
            print("|".join(line))


if __name__ == "__main__":
    main()
