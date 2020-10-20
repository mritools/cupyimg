from __future__ import absolute_import
from __future__ import print_function

import functools
import traceback
import unittest
import warnings

import numpy

import cupy
from cupy.testing import array

import cupyx.scipy.sparse
import cupyimg
import cupyimg.scipy


def _call_func(self, impl, args, kw):
    try:
        result = impl(self, *args, **kw)
        error = None
        tb = None
    except Exception as e:
        tb = e.__traceback__
        if tb.tb_next is None:
            # failed before impl is called, e.g. invalid kw
            raise e
        result = None
        error = e

    return result, error, tb


def _call_func_cupyimg(self, impl, args, kw, name, sp_name, scipy_name):
    assert isinstance(name, str)
    assert sp_name is None or isinstance(sp_name, str)
    assert scipy_name is None or isinstance(scipy_name, str)
    kw = kw.copy()

    # Run cupy
    if sp_name:
        kw[sp_name] = cupyx.scipy.sparse
    if scipy_name:
        kw[scipy_name] = cupyimg.scipy
    kw[name] = cupy
    result, error, tb = _call_func(self, impl, args, kw)
    return result, error, tb


def _call_func_numpy(self, impl, args, kw, name, sp_name, scipy_name):
    assert isinstance(name, str)
    assert sp_name is None or isinstance(sp_name, str)
    assert scipy_name is None or isinstance(scipy_name, str)
    kw = kw.copy()

    # Run numpy
    kw[name] = numpy
    if sp_name:
        import scipy.sparse

        kw[sp_name] = scipy.sparse
    if scipy_name:
        import scipy

        kw[scipy_name] = scipy
    result, error, tb = _call_func(self, impl, args, kw)
    return result, error, tb


def _call_func_numpy_cupy(self, impl, args, kw, name, sp_name, scipy_name):
    # Run cupy
    cupy_result, cupy_error, cupy_tb = _call_func_cupyimg(
        self, impl, args, kw, name, sp_name, scipy_name
    )

    # Run numpy
    numpy_result, numpy_error, numpy_tb = _call_func_numpy(
        self, impl, args, kw, name, sp_name, scipy_name
    )

    return (
        cupy_result,
        cupy_error,
        cupy_tb,
        numpy_result,
        numpy_error,
        numpy_tb,
    )


_numpy_errors = [
    AttributeError,
    Exception,
    IndexError,
    TypeError,
    ValueError,
    NotImplementedError,
    DeprecationWarning,
    numpy.AxisError,
    numpy.linalg.LinAlgError,
]


def _check_numpy_cupy_error_compatible(cupy_error, numpy_error):
    """Checks if try/except blocks are equivalent up to public error classes"""

    return all(
        [
            isinstance(cupy_error, err) == isinstance(numpy_error, err)
            for err in _numpy_errors
        ]
    )


def _fail_test_with_unexpected_errors(
    testcase, msg_format, cupy_error, cupy_tb, numpy_error, numpy_tb
):
    # Fails the test due to unexpected errors raised from the test.
    # msg_format may include format placeholders:
    # '{cupy_error}' '{cupy_tb}' '{numpy_error}' '{numpy_tb}'

    msg = msg_format.format(
        cupy_error="".join(str(cupy_error)),
        cupy_tb="".join(traceback.format_tb(cupy_tb)),
        numpy_error="".join(str(numpy_error)),
        numpy_tb="".join(traceback.format_tb(numpy_tb)),
    )

    # Fail the test with the traceback of the error (for pytest --pdb)
    try:
        testcase.fail(msg)
    except AssertionError as e:
        raise e.with_traceback(cupy_tb or numpy_tb)
    assert False  # never reach


def _check_cupy_numpy_error(
    self, cupy_error, cupy_tb, numpy_error, numpy_tb, accept_error=False
):
    # Skip the test if both raised SkipTest.
    if isinstance(cupy_error, unittest.SkipTest) and isinstance(
        numpy_error, unittest.SkipTest
    ):
        if cupy_error.args != numpy_error.args:
            raise AssertionError(
                "Both numpy and cupy were skipped but with different causes."
            )
        raise numpy_error  # reraise SkipTest

    # For backward compatibility
    if accept_error is True:
        accept_error = Exception
    elif not accept_error:
        accept_error = ()
    # TODO(oktua): expected_regexp like numpy.testing.assert_raises_regex
    if cupy_error is None and numpy_error is None:
        self.fail("Both cupy and numpy are expected to raise errors, but not")
    elif cupy_error is None:
        _fail_test_with_unexpected_errors(
            self,
            "Only numpy raises error\n\n{numpy_tb}{numpy_error}",
            None,
            None,
            numpy_error,
            numpy_tb,
        )
    elif numpy_error is None:
        _fail_test_with_unexpected_errors(
            self,
            "Only cupy raises error\n\n{cupy_tb}{cupy_error}",
            cupy_error,
            cupy_tb,
            None,
            None,
        )

    elif not _check_numpy_cupy_error_compatible(cupy_error, numpy_error):
        _fail_test_with_unexpected_errors(
            self,
            """Different types of errors occurred

cupy
{cupy_tb}{cupy_error}

numpy
{numpy_tb}{numpy_error}
""",
            cupy_error,
            cupy_tb,
            numpy_error,
            numpy_tb,
        )

    elif not (
        isinstance(cupy_error, accept_error)
        and isinstance(numpy_error, accept_error)
    ):
        _fail_test_with_unexpected_errors(
            self,
            """Both cupy and numpy raise exceptions

cupy
{cupy_tb}{cupy_error}

numpy
{numpy_tb}{numpy_error}
""",
            cupy_error,
            cupy_tb,
            numpy_error,
            numpy_tb,
        )


def _make_positive_mask(self, impl, args, kw, name, sp_name, scipy_name):
    # Returns a mask of output arrays that indicates valid elements for
    # comparison. See the comment at the caller.
    ks = [k for k, v in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = numpy.intp
    result, error, tb = _call_func_cupyimg(
        self, impl, args, kw, name, sp_name, scipy_name
    )
    assert error is None
    return cupy.asnumpy(result) >= 0


def _contains_signed_and_unsigned(kw):
    vs = set(kw.values())
    return any(d in vs for d in _unsigned_dtypes) and any(
        d in vs for d in _float_dtypes + _signed_dtypes
    )


def _make_decorator(
    check_func,
    name,
    type_check,
    contiguous_check,
    accept_error,
    sp_name=None,
    scipy_name=None,
    check_sparse_format=True,
):
    assert isinstance(name, str)
    assert sp_name is None or isinstance(sp_name, str)
    assert scipy_name is None or isinstance(scipy_name, str)

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # Run cupy and numpy
            (
                cupy_result,
                cupy_error,
                cupy_tb,
                numpy_result,
                numpy_error,
                numpy_tb,
            ) = _call_func_numpy_cupy(
                self, impl, args, kw, name, sp_name, scipy_name
            )
            assert cupy_result is not None or cupy_error is not None
            assert numpy_result is not None or numpy_error is not None

            # Check errors raised
            if cupy_error or numpy_error:
                _check_cupy_numpy_error(
                    self,
                    cupy_error,
                    cupy_tb,
                    numpy_error,
                    numpy_tb,
                    accept_error=accept_error,
                )
                return

            # Check returned arrays

            if not isinstance(cupy_result, (tuple, list)):
                cupy_result = (cupy_result,)
            if not isinstance(numpy_result, (tuple, list)):
                numpy_result = (numpy_result,)

            assert len(cupy_result) == len(numpy_result)

            # Check types
            cupy_numpy_result_ndarrays = [
                _convert_output_to_ndarray(
                    cupy_r, numpy_r, sp_name, check_sparse_format
                )
                for cupy_r, numpy_r in zip(cupy_result, numpy_result)
            ]

            # Check dtypes
            if type_check:
                for cupy_r, numpy_r in cupy_numpy_result_ndarrays:
                    assert cupy_r.dtype == numpy_r.dtype

            # Check contiguities
            if contiguous_check:
                for cupy_r, numpy_r in zip(cupy_result, numpy_result):
                    if isinstance(numpy_r, numpy.ndarray):
                        if (
                            numpy_r.flags.c_contiguous
                            and not cupy_r.flags.c_contiguous
                        ):
                            raise AssertionError(
                                "The state of c_contiguous flag is false. "
                                "(cupy_result:{} numpy_result:{})".format(
                                    cupy_r.flags.c_contiguous,
                                    numpy_r.flags.c_contiguous,
                                )
                            )
                        if (
                            numpy_r.flags.f_contiguous
                            and not cupy_r.flags.f_contiguous
                        ):
                            raise AssertionError(
                                "The state of f_contiguous flag is false. "
                                "(cupy_result:{} numpy_result:{})".format(
                                    cupy_r.flags.f_contiguous,
                                    numpy_r.flags.f_contiguous,
                                )
                            )

            # Check shapes
            for cupy_r, numpy_r in cupy_numpy_result_ndarrays:
                assert cupy_r.shape == numpy_r.shape

            # Check item values
            for cupy_r, numpy_r in cupy_numpy_result_ndarrays:
                # Behavior of assigning a negative value to an unsigned integer
                # variable is undefined.
                # nVidia GPUs and Intel CPUs behave differently.
                # To avoid this difference, we need to ignore dimensions whose
                # values are negative.

                skip = False
                if (
                    _contains_signed_and_unsigned(kw)
                    and cupy_r.dtype in _unsigned_dtypes
                ):
                    mask = _make_positive_mask(
                        self, impl, args, kw, name, sp_name, scipy_name
                    )
                    if cupy_r.shape == ():
                        skip = (mask == 0).all()
                    else:
                        cupy_r = cupy_r[mask].get()
                        numpy_r = numpy_r[mask]

                if not skip:
                    check_func(cupy_r, numpy_r)

        return test_func

    return decorator


def _convert_output_to_ndarray(c_out, n_out, sp_name, check_sparse_format):
    """Checks type of cupy/numpy results and returns cupy/numpy ndarrays.

    Args:
        c_out (cupy.ndarray, cupyx.scipy.sparse matrix, cupy.poly1d or scalar):
            cupy result
        n_out (numpy.ndarray, scipy.sparse matrix, numpy.poly1d or scalar):
            numpy result
        sp_name(str or None): Argument name whose value is either
            ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
            argument is given for the modules.
        check_sparse_format (bool): If ``True``, consistency of format of
            sparse matrix is also checked. Default is ``True``.

    Returns:
        The tuple of cupy.ndarray and numpy.ndarray.
    """
    if sp_name is not None and cupyx.scipy.sparse.issparse(c_out):
        # Sparse output case.
        import scipy.sparse

        assert scipy.sparse.issparse(n_out)
        if check_sparse_format:
            assert c_out.format == n_out.format
        return c_out.A, n_out.A
    if isinstance(c_out, cupy.ndarray) and isinstance(
        n_out, (numpy.ndarray, numpy.generic)
    ):
        # ndarray output case.
        return c_out, n_out
    if isinstance(c_out, cupy.poly1d) and isinstance(n_out, numpy.poly1d):
        # poly1d output case.
        assert c_out.variable == n_out.variable
        return c_out.coeffs, n_out.coeffs
    if isinstance(c_out, numpy.generic) and isinstance(n_out, numpy.generic):
        # numpy scalar output case.
        return c_out, n_out
    if numpy.isscalar(c_out) and numpy.isscalar(n_out):
        # python scalar output case.
        return cupy.array(c_out), numpy.array(n_out)
    raise AssertionError(
        "numpy and cupy returns different type of return value:\n"
        "cupy: {}\nnumpy: {}".format(type(c_out), type(n_out))
    )


def numpy_cupyimg_allclose(
    rtol=1e-7,
    atol=0,
    err_msg="",
    verbose=True,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
    contiguous_check=True,
    *,
    _check_sparse_format=True,
):
    """Decorator that checks NumPy results and CuPy ones are close.

    Args:
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.
         contiguous_check(bool): If ``True``, consistency of contiguity is
             also checked.

    Decorated test fixture is required to return the arrays whose values are
    close between ``numpy`` case and ``cupy`` case.
    For example, this test case checks ``numpy.zeros`` and ``cupy.zeros``
    should return same value.

    >>> import unittest
    >>> from cupy import testing
    >>> @testing.gpu
    ... class TestFoo(unittest.TestCase):
    ...
    ...     @testing.numpy_cupyimg_allclose()
    ...     def test_foo(self, xp):
    ...         # ...
    ...         # Prepare data with xp
    ...         # ...
    ...
    ...         xp_result = xp.zeros(10)
    ...         return xp_result

    .. seealso:: :func:`cupy.testing.assert_allclose`
    """

    def check_func(c, n):
        array.assert_allclose(c, n, rtol, atol, err_msg, verbose)

    return _make_decorator(
        check_func,
        name,
        type_check,
        contiguous_check,
        accept_error,
        sp_name,
        scipy_name,
        _check_sparse_format,
    )


def numpy_cupyimg_array_almost_equal(
    decimal=6,
    err_msg="",
    verbose=True,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
):
    """Decorator that checks NumPy results and CuPy ones are almost equal.

    Args:
         decimal(int): Desired precision.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`cupy.testing.assert_array_almost_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_almost_equal`
    """

    def check_func(x, y):
        array.assert_array_almost_equal(x, y, decimal, err_msg, verbose)

    return _make_decorator(
        check_func, name, type_check, False, accept_error, sp_name, scipy_name
    )


def numpy_cupyimg_array_almost_equal_nulp(
    nulp=1,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
):
    """Decorator that checks results of NumPy and CuPy are equal w.r.t. spacing.

    Args:
         nulp(int): The maximum number of unit in the last place for tolerance.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True``, all error types are acceptable.
             If it is ``False``, no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`cupy.testing.assert_array_almost_equal_nulp`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_almost_equal_nulp`
    """  # NOQA

    def check_func(x, y):
        array.assert_array_almost_equal_nulp(x, y, nulp)

    return _make_decorator(
        check_func,
        name,
        type_check,
        False,
        accept_error,
        sp_name,
        scipy_name=None,
    )


def numpy_cupyimg_array_max_ulp(
    maxulp=1,
    dtype=None,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
):
    """Decorator that checks results of NumPy and CuPy ones are equal w.r.t. ulp.

    Args:
         maxulp(int): The maximum number of units in the last place
             that elements of resulting two arrays can differ.
         dtype(numpy.dtype): Data-type to convert the resulting
             two array to if given.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`assert_array_max_ulp`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_max_ulp`

    """  # NOQA

    def check_func(x, y):
        array.assert_array_max_ulp(x, y, maxulp, dtype)

    return _make_decorator(
        check_func, name, type_check, False, accept_error, sp_name, scipy_name
    )


def numpy_cupyimg_array_equal(
    err_msg="",
    verbose=True,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
    strides_check=False,
):
    """Decorator that checks NumPy results and CuPy ones are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.
         strides_check(bool): If ``True``, consistency of strides is also
             checked.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_cupyimg_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_equal`
    """

    def check_func(x, y):
        array.assert_array_equal(x, y, err_msg, verbose, strides_check)

    return _make_decorator(
        check_func, name, type_check, False, accept_error, sp_name, scipy_name
    )


def numpy_cupyimg_array_list_equal(
    err_msg="", verbose=True, name="xp", sp_name=None, scipy_name=None
):
    """Decorator that checks the resulting lists of NumPy and CuPy's one are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are appended
             to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same list of arrays
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_list_equal`
    """  # NOQA
    warnings.warn(
        "numpy_cupy_array_list_equal is deprecated."
        " Use numpy_cupy_array_equal instead.",
        DeprecationWarning,
    )

    def check_func(x, y):
        array.assert_array_equal(x, y, err_msg, verbose)

    return _make_decorator(
        check_func, name, False, False, False, sp_name, scipy_name
    )


def numpy_cupyimg_array_less(
    err_msg="",
    verbose=True,
    name="xp",
    type_check=True,
    accept_error=False,
    sp_name=None,
    scipy_name=None,
):
    """Decorator that checks the CuPy result is less than NumPy result.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the smaller array
    when ``xp`` is ``cupy`` than the one when ``xp`` is ``numpy``.

    .. seealso:: :func:`cupy.testing.assert_array_less`
    """

    def check_func(x, y):
        array.assert_array_less(x, y, err_msg, verbose)

    return _make_decorator(
        check_func, name, type_check, False, accept_error, sp_name, scipy_name
    )


def numpy_cupyimg_equal(name="xp", sp_name=None, scipy_name=None):
    """Decorator that checks NumPy results are equal to CuPy ones.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyimg.sciyp.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.

    Decorated test fixture is required to return the same results
    even if ``xp`` is ``numpy`` or ``cupy``.
    """

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # Run cupy and numpy
            (
                cupy_result,
                cupy_error,
                cupy_tb,
                numpy_result,
                numpy_error,
                numpy_tb,
            ) = _call_func_numpy_cupy(
                self, impl, args, kw, name, sp_name, scipy_name
            )

            if cupy_result != numpy_result:
                message = """Results are not equal:
cupy: %s
numpy: %s""" % (
                    str(cupy_result),
                    str(numpy_result),
                )
                raise AssertionError(message)

        return test_func

    return decorator


def numpy_cupyimg_raises(
    name="xp", sp_name=None, scipy_name=None, accept_error=Exception
):
    """Decorator that checks the NumPy and CuPy throw same errors.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupyx.scipy.sparse`` module. If ``None``, no
             argument is given for the modules.
         scipy_name(str or None): Argument name whose value is either ``scipy``
             or ``cupyimg.scipy`` module. If ``None``, no argument is given for
             the modules.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.

    Decorated test fixture is required throw same errors
    even if ``xp`` is ``numpy`` or ``cupy``.
    """
    warnings.warn(
        "cupy.testing.numpy_cupy_raises is deprecated.", DeprecationWarning
    )

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            # Run cupy and numpy
            (
                cupy_result,
                cupy_error,
                cupy_tb,
                numpy_result,
                numpy_error,
                numpy_tb,
            ) = _call_func_numpy_cupy(
                self, impl, args, kw, name, sp_name, scipy_name
            )

            _check_cupy_numpy_error(
                self,
                cupy_error,
                cupy_tb,
                numpy_error,
                numpy_tb,
                accept_error=accept_error,
            )

        return test_func

    return decorator


def for_dtypes(dtypes, name="dtype"):
    """Decorator for parameterized dtype test.

    Args:
         dtypes(list of dtypes): dtypes to be tested.
         name(str): Argument name to which specified dtypes are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.
    """

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for dtype in dtypes:
                try:
                    kw[name] = numpy.dtype(dtype).type
                    impl(self, *args, **kw)
                except unittest.SkipTest as e:
                    print("skipped: {} = {} ({})".format(name, dtype, e))
                except Exception:
                    print(name, "is", dtype)
                    raise

        return test_func

    return decorator


_complex_dtypes = (numpy.complex64, numpy.complex128)
_regular_float_dtypes = (numpy.float64, numpy.float32)
_float_dtypes = _regular_float_dtypes + (numpy.float16,)
_signed_dtypes = tuple(numpy.dtype(i).type for i in "bhilq")
_unsigned_dtypes = tuple(numpy.dtype(i).type for i in "BHILQ")
_int_dtypes = _signed_dtypes + _unsigned_dtypes
_int_bool_dtypes = _int_dtypes + (numpy.bool_,)
_regular_dtypes = _regular_float_dtypes + _int_bool_dtypes
_dtypes = _float_dtypes + _int_bool_dtypes
