"""CuPy Extensions

This project contains CuPy-based implementations of functions from NumPy,
SciPy and scikit-image that are not currently available in CuPy itself.

Most functions are not provided via the top level-import. Instead, individual
subpackages should be imported instead.

Subpackages
-----------
numpy
    Functions from NumPy which are not available via CuPy.
scipy
    Functions from SciPy which are not available via CuPy.
skimage
    Functions from scikit-image.

Additional documentation and usage examples for the functions can be found
at the main documentation pages of the various packges:

"""

import cupy

try:
    memoize = cupy.util.memoize
except AttributeError:
    memoize = cupy.memoize

from ._misc import convolve_separable  # noqa
from .version import __version__  # noqa


del cupy
