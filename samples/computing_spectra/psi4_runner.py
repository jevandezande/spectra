import numpy as np
from psi4 import core
from psi4.driver import freq, geometry, opt

core.IOManager.shared_object()


def run_ir(geom: str, functional: str, basis_set: str) -> tuple[np.ndarray, np.ndarray]:
    geometry(
        """
H
O   1   1
H   2   1   1   104.5"""
    )

    opt("BP86-D3/def2-SVP")
    freq("BP86-D3/def2-SVP")

    energies = np.arange(10)
    intensities = np.arange(10)

    return energies, intensities
