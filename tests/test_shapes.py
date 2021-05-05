import numpy as np
from numpy.testing import assert_almost_equal as aae

from spectra.shapes import gaussian


def setup():
    pass


def teardown():
    pass


def test_gaussian():
    assert np.isnan(gaussian(0, 0, np.arange(1)))[0]

    aae(1, gaussian(0, 1, np.arange(1)))
    aae(1, gaussian(0, 1e-9, np.arange(1)))
    aae(1, gaussian(0, 1e99, np.arange(1)))
    aae(0, gaussian(1e99, 1, np.arange(1)))

    aae(1, gaussian(1, 1e99, np.arange(1)))
    aae(0, gaussian(1, 1e-9, np.arange(1)))

    aae(0.60653066, gaussian(1, 1, np.arange(1)))
    aae([0.60653066, 1, 0.60653066], gaussian(1, 1, np.arange(3)))
    aae([0.1353353, 0.60653066, 1, 0.60653066, 0.1353353], gaussian(2, 1, np.arange(5)))
