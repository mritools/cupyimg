from __future__ import absolute_import
from __future__ import print_function

import functools
import sys
import traceback
import unittest

import numpy

import cupy
from cupy.testing import array
import cupyx.scipy.sparse
import cupyimg
import cupyimg.scipy


def _call_func(self, impl, args, kw):
    try:
        result = impl(self, *args, **kw)
        assert result is not None
        error = None
        tb_str = None
    except Exception as e:
        _, _, tb = sys.exc_info()  # e.__traceback__ is py3 only
        if tb.tb_next is None:
            # failed before impl is called, e.g. invalid kw
            raise e
        result = None
        error = e
        tb_str = traceback.format_exc()

    return result, error, tb_str


def _get_numpy_errors():
    numpy_version = numpy.lib.NumpyVersion(numpy.__version__)

    errors = [
        AttributeError,
        Exception,
        IndexError,
        TypeError,
        ValueError,
        NotImplementedError,
        DeprecationWarning,
    ]
    if numpy_version >= "1.13.0":
        errors.append(numpy.AxisError)
    if numpy_version >= "1.15.0":
        errors.append(numpy.linalg.LinAlgError)

    return errors


_numpy_errors = _get_numpy_errors()


def _check_numpy_cupyimg_error_compatible(cupy_error, numpy_error):
    """Checks if try/except blocks are equivalent up to public error classes
    """

    errors = _numpy_errors

    # Prior to NumPy version 1.13.0, NumPy raises either `ValueError` or
    # `IndexError` instead of `numpy.AxisError`.
    numpy_axis_error = getattr(numpy, "AxisError", None)
    cupy_axis_error = cupy.core._errors._AxisError
    if isinstance(cupy_error, cupy_axis_error) and numpy_axis_error is None:
        if not isinstance(numpy_error, (ValueError, IndexError)):
            return False
        errors = list(set(errors) - set([IndexError, ValueError]))

    return all(
        [
            isinstance(cupy_error, err) == isinstance(numpy_error, err)
            for err in errors
        ]
    )


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
        self.fail("Only numpy raises error\n\n" + numpy_tb)
    elif numpy_error is None:
        self.fail("Only cupy raises error\n\n" + cupy_tb)

    elif not _check_numpy_cupyimg_error_compatible(cupy_error, numpy_error):
        msg = """Different types of errors occurred

cupy
%s
numpy
%s
""" % (
            cupy_tb,
            numpy_tb,
        )
        self.fail(msg)

    elif not (
        isinstance(cupy_error, accept_error)
        and isinstance(numpy_error, accept_error)
    ):
        msg = """Both cupy and numpy raise exceptions

cupy
%s
numpy
%s
""" % (
            cupy_tb,
            numpy_tb,
        )
        self.fail(msg)


def _make_positive_mask(self, impl, args, kw):
    ks = [k for k, v in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = numpy.intp
    return cupy.asnumpy(impl(self, *args, **kw)) >= 0


def _contains_signed_and_unsigned(kw):
    vs = set(kw.values())
    return any(d in vs for d in _unsigned_dtypes) and any(
        d in vs for d in _float_dtypes + _signed_dtypes
    )


def _make_decorator(
    check_func, name, type_check, accept_error, sp_name=None, scipy_name=None
):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if sp_name:
                kw[sp_name] = cupyx.scipy.sparse
            if scipy_name:
                kw[scipy_name] = cupyimg.scipy
            kw[name] = cupy
            cupy_result, cupy_error, cupy_tb = _call_func(self, impl, args, kw)

            kw[name] = numpy
            if sp_name:
                import scipy.sparse

                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy

                kw[scipy_name] = scipy
            numpy_result, numpy_error, numpy_tb = _call_func(
                self, impl, args, kw
            )

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

            assert cupy_result.shape == numpy_result.shape

            # Behavior of assigning a negative value to an unsigned integer
            # variable is undefined.
            # nVidia GPUs and Intel CPUs behave differently.
            # To avoid this difference, we need to ignore dimensions whose
            # values are negative.
            skip = False
            if (
                _contains_signed_and_unsigned(kw)
                and cupy_result.dtype in _unsigned_dtypes
            ):
                mask = _make_positive_mask(self, impl, args, kw)
                if cupy_result.shape == ():
                    skip = (mask == 0).all()
                else:
                    cupy_result = cupy.asnumpy(cupy_result[mask])
                    numpy_result = cupy.asnumpy(numpy_result[mask])

            if not skip:
                check_func(cupy_result, numpy_result)
            if type_check:
                assert cupy_result.dtype == numpy_result.dtype

        return test_func

    return decorator


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
        c_array = c
        n_array = n
        if sp_name is not None:
            import scipy.sparse

            if cupyx.scipy.sparse.issparse(c):
                c_array = c.A
            if scipy.sparse.issparse(n):
                n_array = n.A
        array.assert_allclose(c_array, n_array, rtol, atol, err_msg, verbose)
        if contiguous_check and isinstance(n, numpy.ndarray):
            if n.flags.c_contiguous and not c.flags.c_contiguous:
                raise AssertionError(
                    "The state of c_contiguous flag is false. "
                    "(cupy_result:{} numpy_result:{})".format(
                        c.flags.c_contiguous, n.flags.c_contiguous
                    )
                )
            if n.flags.f_contiguous and not c.flags.f_contiguous:
                raise AssertionError(
                    "The state of f_contiguous flag is false. "
                    "(cupy_result:{} numpy_result:{})".format(
                        c.flags.f_contiguous, n.flags.f_contiguous
                    )
                )

    return _make_decorator(
        check_func, name, type_check, accept_error, sp_name, scipy_name
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
        check_func, name, type_check, accept_error, sp_name, scipy_name
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
        check_func, name, type_check, accept_error, sp_name, scipy_name=None
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
        check_func, name, type_check, accept_error, sp_name, scipy_name
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
        if sp_name is not None:
            import scipy.sparse

            if cupyx.scipy.sparse.issparse(x):
                x = x.A
            if scipy.sparse.issparse(y):
                y = y.A

        array.assert_array_equal(x, y, err_msg, verbose, strides_check)

    return _make_decorator(
        check_func, name, type_check, accept_error, sp_name, scipy_name
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

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if sp_name:
                kw[sp_name] = cupyx.scipy.sparse
            if scipy_name:
                kw[scipy_name] = cupyimg.scipy
            kw[name] = cupy
            x = impl(self, *args, **kw)

            if sp_name:
                import scipy.sparse

                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy

                kw[scipy_name] = scipy
            kw[name] = numpy
            y = impl(self, *args, **kw)
            assert x is not None
            assert y is not None
            array.assert_array_list_equal(x, y, err_msg, verbose)

        return test_func

    return decorator


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
        check_func, name, type_check, accept_error, sp_name, scipy_name
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
            if sp_name:
                kw[sp_name] = cupyx.scipy.sparse
            if scipy_name:
                kw[scipy_name] = cupyimg.scipy
            kw[name] = cupy
            cupy_result = impl(self, *args, **kw)

            if sp_name:
                import scipy.sparse

                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy

                kw[scipy_name] = scipy
            kw[name] = numpy
            numpy_result = impl(self, *args, **kw)

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

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if sp_name:
                kw[sp_name] = cupyx.scipy.sparse
            if scipy_name:
                kw[scipy_name] = cupyimg.scipy
            kw[name] = cupy
            try:
                impl(self, *args, **kw)
                cupy_error = None
                cupy_tb = None
            except Exception as e:
                cupy_error = e
                cupy_tb = traceback.format_exc()

            if sp_name:
                import scipy.sparse

                kw[sp_name] = scipy.sparse
            if scipy_name:
                import scipy

                kw[scipy_name] = scipy
            kw[name] = numpy
            try:
                impl(self, *args, **kw)
                numpy_error = None
                numpy_tb = None
            except Exception as e:
                numpy_error = e
                numpy_tb = traceback.format_exc()

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


_complex_dtypes = (numpy.complex64, numpy.complex128)
_regular_float_dtypes = (numpy.float64, numpy.float32)
_float_dtypes = _regular_float_dtypes + (numpy.float16,)
_signed_dtypes = tuple(numpy.dtype(i).type for i in "bhilq")
_unsigned_dtypes = tuple(numpy.dtype(i).type for i in "BHILQ")
_int_dtypes = _signed_dtypes + _unsigned_dtypes
_int_bool_dtypes = _int_dtypes + (numpy.bool_,)
_regular_dtypes = _regular_float_dtypes + _int_bool_dtypes
_dtypes = _float_dtypes + _int_bool_dtypes
