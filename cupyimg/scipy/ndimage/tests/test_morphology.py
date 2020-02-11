import cupy
from pytest import raises as assert_raises

import cupyimg.scipy.ndimage as sndi


def test_binary_erosion_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = cupy.ones([1])
    assert_raises(TypeError, sndi.binary_erosion, data, iterations=0.5)
    assert_raises(TypeError, sndi.binary_erosion, data, iterations=1.5)


def test_binary_dilation_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = cupy.ones([1])
    assert_raises(TypeError, sndi.binary_dilation, data, iterations=0.5)
    assert_raises(TypeError, sndi.binary_dilation, data, iterations=1.5)


def test_binary_opening_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = cupy.ones([1])
    assert_raises(TypeError, sndi.binary_opening, data, iterations=0.5)
    assert_raises(TypeError, sndi.binary_opening, data, iterations=1.5)


def test_binary_closing_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = cupy.ones([1])
    assert_raises(TypeError, sndi.binary_closing, data, iterations=0.5)
    assert_raises(TypeError, sndi.binary_closing, data, iterations=1.5)


def test_binary_closing_noninteger_brute_force_passes_when_true():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = cupy.ones([1])

    assert sndi.binary_erosion(
        data, iterations=2, brute_force=1.5
    ) == sndi.binary_erosion(data, iterations=2, brute_force=bool(1.5))
    # TODO: restore this test case if brute_force=False case is implemented
    # assert sndi.binary_erosion(
    #     data, iterations=2, brute_force=0.0
    # ) == sndi.binary_erosion(data, iterations=2, brute_force=bool(0.0))
