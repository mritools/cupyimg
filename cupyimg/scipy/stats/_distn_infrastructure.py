import cupy as cp
import numpy as np

from cupyimg.scipy.special import entr, rel_entr


def entropy(pk, qk=None, base=None, axis=0):
    """Calculate the entropy of a distribution for given probability values.

    If only probabilities `pk` are given, the entropy is calculated as
    ``S = -sum(pk * log(pk), axis=axis)``.

    If `qk` is not None, then compute the Kullback-Leibler divergence
    ``S = sum(pk * log(pk / qk), axis=axis)``.

    This routine will normalize `pk` and `qk` if they don't sum to 1.

    Parameters
    ----------
    pk : sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk : sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base : float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis: int, optional
        The axis along which the entropy is calculated. Default is 0.

    Returns
    -------
    S : float
        The calculated entropy.

    Examples
    --------

    >>> from scipy.stats import entropy

    Bernoulli trial with different p.
    The outcome of a fair coin is the most uncertain:

    >>> entropy([1/2, 1/2], base=2)
    1.0

    The outcome of a biased coin is less uncertain:

    >>> entropy([9/10, 1/10], base=2)
    0.46899559358928117

    Relative entropy:

    >>> entropy([1/2, 1/2], qk=[9/10, 1/10])
    0.5108256237659907

    """
    pk = cp.asarray(pk)
    pk = 1.0 * pk / cp.sum(pk, axis=axis, keepdims=True)
    if qk is None:
        vec = entr(pk)
    else:
        qk = cp.asarray(qk)
        if qk.shape != pk.shape:
            raise ValueError("qk and pk must have same shape.")
        qk = 1.0 * qk / cp.sum(qk, axis=axis, keepdims=True)
        vec = rel_entr(pk, qk)
    S = cp.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S
