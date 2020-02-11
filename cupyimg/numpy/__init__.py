"""Implementations of functions from the NumPy API.

The convolve and correlate as defined here only perform the computations using
floating point (or complex) dtypes. Specifically only:
np.float32, np.float64, np.complex64, np.complex128

Other dtypes can be used for the input, but the output will be the nearest
floating type.

"""

from .core.fromnumeric import ndim
from .core.numeric import convolve, correlate
from .lib import gradient

__all__ = ["convolve", "correlate", "gradient", "ndim"]
