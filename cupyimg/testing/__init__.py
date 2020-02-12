"""These utilities are copied with minor modification from cupy.testing

The only changes are to try functions from cupyimg.scipy instead of cupyx.scipy
as used in a number of the tests within the cupyimg.scipy folder.
"""

from cupyimg.testing.helper import numpy_cupyimg_allclose  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_array_almost_equal  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_array_almost_equal_nulp  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_array_equal  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_array_less  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_array_list_equal  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_array_max_ulp  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_equal  # NOQA
from cupyimg.testing.helper import numpy_cupyimg_raises  # NOQA
