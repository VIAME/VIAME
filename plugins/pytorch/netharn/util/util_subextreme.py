# -*- coding: utf-8 -*-
import numpy as np
import ubelt as ub  # NOQA


def argsubmax(ydata, xdata=None):
    """
    Finds a single submaximum value to subindex accuracy.
    If xdata is not specified, submax_x is a fractional index.
    Otherwise, submax_x is sub-xdata (essentially doing the index interpolation
    for you)

    Example:
        >>> from .util.util_subextreme import *  # NOQA
        >>> ydata = [ 0,  1,  2, 1.5,  0]
        >>> xdata = [00, 10, 20,  30, 40]
        >>> result1 = argsubmax(ydata, xdata=None)
        >>> result2 = argsubmax(ydata, xdata=xdata)
        >>> result = ub.repr2([result1, result2], precision=4, nl=1, nobr=1)
        >>> print(result)
        (2.1667, 2.0208),
        (21.6667, 2.0208),

    Example:
        >>> hist_ = np.array([0, 1, 2, 3, 4])
        >>> centers = None
        >>> maxima_thresh=None
        >>> argsubmax(hist_)
        (4.0, 4.0)
    """
    if len(ydata) == 0:
        raise IndexError('zero length array')
    ydata = np.asarray(ydata)
    xdata = None if xdata is None else np.asarray(xdata)
    submaxima_x, submaxima_y = argsubmaxima(ydata, centers=xdata)
    idx = submaxima_y.argmax()
    submax_y = submaxima_y[idx]
    submax_x = submaxima_x[idx]
    return submax_x, submax_y


def argsubmaxima(hist, centers=None, maxima_thresh=None, _debug=False):
    r"""
    Determines approximate maxima values to subindex accuracy.

    Args:
        hist_ (ndarray): ydata, histogram frequencies
        centers (ndarray): xdata, histogram labels
        maxima_thresh (float): cutoff point for labeing a value as a maxima

    Returns:
        tuple: (submaxima_x, submaxima_y)

    Example:
        >>> maxima_thresh = .8
        >>> hist = np.array([6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> (submaxima_x, submaxima_y) = argsubmaxima(hist, centers, maxima_thresh)
        >>> result = str((submaxima_x, submaxima_y))
        >>> print(result)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> # TODO? Port from plottool?
        >>> #pt.draw_hist_subbin_maxima(hist, centers)
        >>> #pt.show_if_requested()
        (array([ 3.0318792]), array([ 37.19208239]))
    """
    maxima_x, maxima_y, argmaxima = _hist_argmaxima(hist, centers, maxima_thresh=maxima_thresh)
    argmaxima = np.asarray(argmaxima)
    if _debug:
        print('Argmaxima: ')
        print(' * maxima_x = %r' % (maxima_x))
        print(' * maxima_y = %r' % (maxima_y))
        print(' * argmaxima = %r' % (argmaxima))
    flags = (argmaxima == 0) | (argmaxima == len(hist) - 1)
    argmaxima_ = argmaxima[~flags]
    submaxima_x_, submaxima_y_ = _interpolate_submaxima(argmaxima_, hist, centers)
    if np.any(flags):
        endpts = argmaxima[flags]
        submaxima_x = (np.hstack([submaxima_x_, centers[endpts]])
                       if centers is not None else
                       np.hstack([submaxima_x_, endpts]))
        submaxima_y = np.hstack([submaxima_y_, hist[endpts]])
    else:
        submaxima_y = submaxima_y_
        submaxima_x = submaxima_x_
    if _debug:
        print('Submaxima: ')
        print(' * submaxima_x = %r' % (submaxima_x))
        print(' * submaxima_y = %r' % (submaxima_y))
    return submaxima_x, submaxima_y


def _hist_argmaxima(hist, centers=None, maxima_thresh=None):
    """
    must take positive only values

    Example:
        >>> # ENABLE_DOCTEST
        >>> maxima_thresh = .8
        >>> hist = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> maxima_x, maxima_y, argmaxima = _hist_argmaxima(hist, centers)
        >>> result = str((maxima_x, maxima_y, argmaxima))
        >>> print(result)
    """
    # FIXME: Not handling general cases
    # [0] index because argrelmaxima returns a tuple
    import scipy.signal
    argmaxima_ = scipy.signal.argrelextrema(hist, np.greater)[0]
    if len(argmaxima_) == 0:
        argmaxima_ = hist.argmax()
    if maxima_thresh is not None:
        # threshold maxima to be within a factor of the maximum
        maxima_y = hist[argmaxima_]
        isvalid = maxima_y > maxima_y.max() * maxima_thresh
        argmaxima = argmaxima_[isvalid]
    else:
        argmaxima = argmaxima_
    maxima_y = hist[argmaxima]
    maxima_x = argmaxima if centers is None else centers[argmaxima]
    return maxima_x, maxima_y, argmaxima


def _interpolate_submaxima(argmaxima, hist_, centers=None):
    r"""
    Args:
        argmaxima (ndarray): indicies into ydata / centers that are argmaxima
        hist_ (ndarray): ydata, histogram frequencies
        centers (ndarray): xdata, histogram labels

    FIXME:
        what happens when argmaxima[i] == len(hist_)

    Example:
        >>> argmaxima = np.array([1, 4, 7])
        >>> hist_ = np.array([    6.73, 8.69, 0.00, 0.00, 34.62, 29.16, 0.00, 0.00, 6.73, 8.69])
        >>> centers = np.array([-0.39, 0.39, 1.18, 1.96,  2.75,  3.53, 4.32, 5.11, 5.89, 6.68])
        >>> submaxima_x, submaxima_y = _interpolate_submaxima(argmaxima, hist_, centers)
        >>> # Populate some internal variables
        >>> argmaxima = np.asarray(argmaxima)
        >>> neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
        >>> y123 = hist_[neighbs]
        >>> x123 = neighbs if centers is None else centers[neighbs]
        >>> coeff_list = [np.polyfit(x123_, y123_, deg=2)
        >>>               for (x123_, y123_) in zip(x123.T, y123.T)]
        >>> res = (submaxima_x, submaxima_y)
        >>> result = ub.repr2(res, nl=1, nobr=1, precision=2, with_dtype=True)
        >>> print(result)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autompl()
        >>> ax = kwplot.figure().gca()
        >>> ax.plot(centers, hist_, '-')
        >>> ax.plot(centers[argmaxima], hist_[argmaxima], 'o', label='argmaxima')
        >>> ax.plot(submaxima_x, submaxima_y, 'b*', markersize=20, label='interp maxima')
        >>> # Extract parabola points
        >>> plt.plot(x123, y123, 'o', label='maxima neighbors')
        >>> xpoints = [np.linspace(x1, x3, 50) for (x1, x2, x3) in x123.T]
        >>> ypoints = [np.polyval(coeff, x_pts) for x_pts, coeff in zip(xpoints, coeff_list)]
        >>> # Draw Submax Parabola
        >>> for x_pts, y_pts in zip(xpoints, ypoints):
        >>>     plt.plot(x_pts, y_pts, 'g--', lw=2)
        >>> kwplot.show_if_requested()
    """
    if len(argmaxima) == 0:
        return [], []
    argmaxima = np.asarray(argmaxima)
    neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
    # flags = (neighbs[2] > (len(hist_) - 1)) | (neighbs[0] < 0)
    # neighbs = np.clip(neighbs, 0, len(hist_) - 1)
    # if np.any(flags):
    #     # Clip out of bounds positions
    #     neighbs[0, flags] = neighbs[1, flags]
    #     neighbs[2, flags] = neighbs[1, flags]
    y123 = hist_[neighbs]
    x123 = neighbs if centers is None else centers[neighbs]
    # if np.any(flags):
    #     # Make symetric values so maxima is found exactly in center
    #     y123[0, flags] = y123[1, flags] - 1
    #     y123[2, flags] = y123[1, flags] - 1
    #     x123[0, flags] = x123[1, flags] - 1
    #     x123[2, flags] = x123[1, flags] - 1
    # Fit parabola around points
    coeff_list = [np.polyfit(x123_, y123_, deg=2)
                  for (x123_, y123_) in zip(x123.T, y123.T)]
    A, B, C = np.vstack(coeff_list).T
    submaxima_x, submaxima_y = _maximum_parabola_point(A, B, C)

    # Check to make sure submaxima is not less than original maxima
    # (can be the case only if the maxima is incorrectly given)
    # In this case just return what the user wanted as the maxima
    maxima_y = y123[1, :]
    invalid = submaxima_y < maxima_y
    if np.any(invalid):
        if centers is not None:
            submaxima_x[invalid] = centers[argmaxima[invalid]]
        else:
            submaxima_x[invalid] = argmaxima[invalid]
        submaxima_y[invalid] = hist_[argmaxima[invalid]]
    return submaxima_x, submaxima_y


def _maximum_parabola_point(A, B, C):
    """ Maximum x point is where the derivative is 0 """
    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    return xv, yv


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.util.util_subextreme all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
