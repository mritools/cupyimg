from skimage._shared.utils import deprecated

from ._daisy import daisy
from ._canny import canny
from .peak import peak_local_max
from .corner import hessian_matrix, hessian_matrix_eigvals
from .masked_register_translation import masked_register_translation


@deprecated(
    alt_func="skimage.registration.phase_cross_correlation",
    removed_version="0.19",
)
def register_translation(
    src_image, target_image, upsample_factor=1, space="real", return_error=True
):
    from ..registration._phase_cross_correlation import (
        phase_cross_correlation as func,
    )

    return func(src_image, target_image, upsample_factor, space, return_error)


__all__ = [
    "canny",
    "daisy",
    "peak_local_max",
    "hessian_matrix",
    "hessian_matrix_eigvals",
    "register_translation",
    "masked_register_translation",
]
