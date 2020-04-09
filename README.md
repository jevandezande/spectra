Spectra
=======

[![Travis build](https://img.shields.io/travis/jevandezande/spectra/master.svg?logo=linux&logoColor=white)](https://travis-ci.org/jevandezande/spectra)
[![Codecov](https://codecov.io/gh/jevandezande/spectra/branch/master/graph/badge.svg)](https://codecov.io/gh/jevandezande/spectra)

Spectra is a utility for analyzing and plotting various 1D spectra seen in
chemistry. It currently supports GC, HPLC, MS, NMR, FTIR, Raman, UV-Vis, and
XRD. Spectra can be be subjected to various algebraic manipulations and
integrated across slices, allowing for the plotting of a peaks progress over
time using `progress()`. Curve fitting can be performed, with intelligent
guesses implemented for a few different spectra types. Various utilities for
plotting Spectra are available, along with a large range of ways to adjust the
output figures.

Currently, only data in CSVs is supported.

Samples
-------
![IR Plot](samples/IR/plots/ir_zsh.svg)
![Reaction Kinetics Plot](samples/reaction_kinetics/plots/reaction_kinetics.svg)

Contributing
------------
Additional contributions and suggestions are welcome. The intent is to produce
a small, easily usable package for producing quality graphics and assisting in
workflows. Speed is important, as `progress()` regularly needs to read and
manipulate 100+ Spectra at a time.
