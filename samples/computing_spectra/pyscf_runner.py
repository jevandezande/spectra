import numpy as np
import pyscf
from pyscf.dft.rks import RKS
from pyscf.prop.freq.rks import Frequency as RKS_Frequency


def run_ir(geom: str, functional: str, basis_set: str) -> tuple[np.ndarray, np.ndarray]:
    wfn = energy(geom, functional, basis_set)

    wfn.mol = optimize(wfn)
    freqs = frequencies(wfn)
    intensities = np.zeros(0)

    return freqs, intensities


def energy(geom: str, functional: str, basis_set: str) -> RKS:
    mol = pyscf.M(atom=geom, basis=basis_set, symmetry=False)

    wfn = mol.RKS()
    wfn.xc = functional
    wfn.run()

    return wfn


def optimize(wfn: RKS) -> pyscf.gto.Mole:
    return pyscf.geomopt.optimize(wfn)


def frequencies(wfn: RKS) -> RKS_Frequency:
    return RKS_Frequency(wfn).run()
