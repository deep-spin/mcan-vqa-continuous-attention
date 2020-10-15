# most of this file from Fabian Pedregosa
# https://github.com/openopt/copt/blob/master/copt/tv_prox.py

import numpy as np
import warnings
try:
    from numba import njit
except ImportError:
    from functools import wraps

    def njit(*args, **kw):
        if len(args) == 1 and len(kw)== 0 and hasattr(args[0], '__call__'):
            func = args[0]
            @wraps(func)
            def inner_function(*args, **kwargs):
                return func(*args, **kwargs)
            return inner_function
        else:
            def inner_function(function):
                @wraps(function)
                def wrapper(*args, **kwargs):
                    return function(*args, **kwargs)
                return wrapper
            return inner_function


@njit
def _prox_tv1d(step_size, input, output):
    """low level function call, no checks are performed"""
    width = input.size + 1
    index_low = np.zeros(width, dtype=np.int32)
    slope_low = np.zeros(width, dtype=input.dtype)
    index_up  = np.zeros(width, dtype=np.int32)
    slope_up  = np.zeros(width, dtype=input.dtype)
    index     = np.zeros(width, dtype=np.int32)
    z         = np.zeros(width, dtype=input.dtype)
    y_low     = np.empty(width, dtype=input.dtype)
    y_up      = np.empty(width, dtype=input.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = input[0] - step_size
    y_up[1] = input[0] + step_size
    incr = 1

    for i in range(2, width):
        y_low[i] = y_low[i-1] + input[(i - 1) * incr]
        y_up[i] = y_up[i-1] + input[(i - 1) * incr]

    y_low[width-1] += step_size
    y_up[width-1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]

    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i]-y_low[i-1]
        while (c_low > s_low+1) and (slope_low[max(s_low, c_low-1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low+1:
                slope_low[c_low] = (y_low[i]-y_low[index_low[c_low-1]]) / (i-index_low[c_low-1])
            else:
                slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])

        slope_up[c_up] = y_up[i]-y_up[i-1]
        while (c_up > s_up+1) and (slope_up[max(c_up-1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i]-y_up[index_up[c_up-1]]) / (i-index_up[c_up-1])
            else:
                slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

        while (c_low == s_low+1) and (c_up > s_up+1) and (slope_low[c_low] >= slope_up[s_up+1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        while (c_up == s_up+1) and (c_low>s_low+1) and (slope_up[c_up]<=slope_low[s_low+1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

    for i in range(1, c_low - s_low + 1):
        index[c+i] = index_low[s_low+i]
        z[c+i] = y_low[index[c+i]]
    c = c + c_low-s_low
    j, i = 0, 1
    while i <= c:
        a = (z[i]-z[i-1]) / (index[i]-index[i-1])
        while j < index[i]:
            output[j * incr] = a
            output[j * incr] = a
            j += 1
        i += 1
    return


@njit
def prox_tv1d_cols(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along columns of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        _prox_tv1d(stepsize, A[:, i], out[:, i])
    return out.ravel()


@njit
def prox_tv1d_rows(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along rows of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        _prox_tv1d(stepsize, A[i, :], out[i, :])
    return out.ravel()


def c_prox_tv2d(step_size, x, n_rows, n_cols, max_iter, tol):
    """
    Douglas-Rachford to minimize a 2-dimensional total variation.
    Reference: https://arxiv.org/abs/1411.0589
    """
    n_features = n_rows * n_cols
    p = np.zeros(n_features)
    q = np.zeros(n_features)

    for it in range(max_iter):
        y = x + p
        y = prox_tv1d_cols(step_size, y, n_rows, n_cols)
        p += x - y
        x = y + q
        x = prox_tv1d_rows(step_size, x, n_rows, n_cols)
        q += y - x

        # check convergence
        accuracy = np.max(np.abs(y - x))
        if accuracy < tol:
            break
    else:
        warnings.warn("prox_tv2d did not converged to desired accuracy\n" +
                      "Accuracy reached: %s" % accuracy)
    return x


def prox_tv2d(w, step_size, n_rows, n_cols, max_iter=500, tol=1e-2):
    """
    Computes the proximal operator of the 2-dimensional total variation operator.
    This solves a problem of the form
         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2
    where TV(x) is the two-dimensional total variation. It does so using the
    Douglas-Rachford algorithm [Barbero and Sra, 2014].
    Parameters
    ----------
    w: array
        vector of coefficients
    step_size: float
        step size (often denoted gamma) in proximal objective function
    max_iter: int
    tol: float
    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    Barbero, Alvaro, and Suvrit Sra. "Modular proximal optimization for
    multidimensional total-variation regularization." arXiv preprint
    arXiv:1411.0589 (2014).
    """

    x = w.copy().astype(np.float64)
    return c_prox_tv2d(step_size, x, n_rows, n_cols, max_iter, tol)


def tv2d_numba(X, max_iter=1000, tol=1e-2):
    n_rows, n_cols = X.shape
    x = X.ravel()
    p = prox_tv2d(x, step_size=.5, n_rows=n_rows, n_cols=n_cols,
                  max_iter=max_iter, tol=tol)
    return p.reshape(n_rows, n_cols)

