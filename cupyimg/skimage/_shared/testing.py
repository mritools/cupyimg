from numpy.testing import (  # noqa
    assert_array_equal, assert_array_almost_equal,
    assert_array_less, assert_array_almost_equal_nulp,
    assert_equal, TestCase, assert_allclose,
    assert_almost_equal, assert_, assert_warns,
    assert_no_warnings)
import pytest
from skimage import data
from ._warnings import expected_warnings  # noqa

skipif = pytest.mark.skipif
xfail = pytest.mark.xfail
parametrize = pytest.mark.parametrize
raises = pytest.raises
fixture = pytest.fixture


def fetch(data_filename):
    """Attempt to fetch data, but if unavailable, skip the tests."""
    try:
        # CuPy Backend: TODO: avoid call to non-public _fetch method
        return data._fetch(data_filename)
    except (ConnectionError, ModuleNotFoundError):
        pytest.skip(f'Unable to download {data_filename}')
