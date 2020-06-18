import cupy as cp

from ..util import img_as_float
from .._shared.utils import check_nD


def _prepare_grayscale_input_2D(image):
    image = cp.squeeze(image)
    check_nD(image, 2)
    return img_as_float(image)
