import numpy as np
import pyscf
from pyscf.dft.rks import RKS
from pyscf.tddft import TDDFT

# from pyscf.prop.freq.rks import Frequency as RKS_Frequency


def run_ir(geom: str, functional: str, basis_set: str) -> tuple[np.ndarray, np.ndarray]:
    wfn = energy(geom, functional, basis_set)

    wfn.mol = optimize(wfn)
    freqs = frequencies(wfn)
    intensities = np.zeros(0)

    return freqs, intensities


def run_tddft(
    geom: str,
    functional: str = "B3LYP",
    basis_set: str = "def2-svp",
    nstates: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    wfn = energy(geom, functional, basis_set)
    mytd = tddft(wfn)

    return mytd.energies, mytd.oscillator_strength(gauge="length")


def energy(geom: str, functional: str, basis_set: str) -> RKS:
    mol = pyscf.M(atom=geom, basis=basis_set, symmetry=False)

    wfn = mol.RKS()
    wfn.xc = functional
    wfn.run()

    return wfn


def optimize(wfn: RKS) -> pyscf.gto.Mole:
    return pyscf.geomopt.optimize(wfn)


def frequencies(wfn: RKS):
    return np.arange(10)


def tddft(wfn: RKS, nstates: int = 10) -> TDDFT:
    mytd = TDDFT(wfn)
    mytd.nstates = nstates
    mytd.kernel()
    mytd.analyze()

    return mytd
