=======
spectra
=======


.. image:: https://img.shields.io/travis/jevandezande/spectra.svg
        :target: https://travis-ci.org/jevandezande/spectra
        :alt: Travis status


.. image:: https://codecov.io/gh/jevandezande/spectra/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/jevandezande/spectra
        :alt: Code coverage status


Spectra is a package for analyzing and plotting various 1D spectra seen in
chemistry. It currently supports GC, HPLC, MS, NMR, FTIR, Raman, UV-Vis, XRD,
and XPS. Spectra can be be subjected to various algebraic manipulations and
integrated across slices, allowing for the plotting of a peaks progress over
time using `progress()`. Curve fitting can be performed, with intelligent
guesses implemented for a few different spectra types. Various utilities for
plotting Spectra are available, along with a large range of ways to adjust the
output figures.

Currently, only data in CSVs is supported.


* Free software: MIT license


Samples
-------
.. image:: samples/IR/plots/ir_zsh.svg
        :width: 800px
        :align: center
        :alt: IR Plot

.. image:: samples/reaction_kinetics/plots/reaction_kinetics.svg
        :width: 800px
        :align: center
        :alt: Reaction Kinetics Plot

Contributing
------------
Additional contributions and suggestions are welcome. The intent is to produce
a small, easily usable package for producing quality graphics and assisting in
workflows. Speed is important, as `progress()` regularly needs to read and
manipulate 100+ Spectra at a time.
