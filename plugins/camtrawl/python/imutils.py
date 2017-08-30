# -*- coding: utf-8 -*-
from __future__ import division, print_function
import itertools as it
import numpy as np
import cv2

try:
    import utool as ut
    print, rrr, profile = ut.inject2(__name__)
except ImportError:
    def profile(func):
        return func


def downsample_average_blocks(img, factor):
    """
    Downsamples by averaging blocks of pixels.  Equivalent to a 2D convolution
    with a uniform matrix with elements `1 / factor ** 2` and a stride of
    `factor`.  Unfortunately scipy doesn't seem to have a strided
    implementation
    """
    dsize = tuple(np.divide(img.shape, factor).astype(np.int)[0:2])
    temp_img = np.zeros(dsize)
    for r, c in it.product(range(factor), range(factor)):
        temp_img += img[r::factor, c::factor]
    new_img = temp_img / (factor ** 2)
    return new_img


def imscale(img, scale):
    """
    Resizes an image by a scale factor.

    Because the result image must have an integer number of pixels, the scale
    factor is rounded, and the rounded scale factor is returned.

    Args:
        dsize (ndarray): an image
        scale (float or tuple): desired floating point scale factor
    """
    dsize = img.shape[0:2][::-1]
    try:
        sx, sy = scale
    except TypeError:
        sx = sy = scale
    w, h = dsize
    new_w = int(round(w * sx))
    new_h = int(round(h * sy))
    new_scale = new_w / w, new_h / h
    new_dsize = (new_w, new_h)

    # new_img = cv2.resize(img, new_dsize, interpolation=cv2.INTER_LANCZOS4)
    new_img = cv2.resize(img, new_dsize, interpolation=cv2.INTER_LINEAR)
    return new_img, new_scale


def to_homog(pts):
    """
    converts [D, :] -> [D + 1, :]

    Args:
        pts (ndarray): in format [D x N] where N is the number of points and
            D is the dimensionality

    Example:
        >>> pts = np.array([[1, 2, 3, 4, 5],
        >>>                 [1, 2, 3, 4, 5]])
        >>> to_homog(pts)
        array([[1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5],
               [1, 1, 1, 1, 1]])
    """
    D, N = pts.shape
    homog_dim = np.ones((1, N), dtype=pts.dtype)
    homog_pts = np.vstack([pts, homog_dim])
    return homog_pts


def from_homog(homog_pts):
    """
    converts [D + 1, :] -> [D, :]

    Args:
        pts (ndarray): in format [(D + 1) x N] where N is the number of points
            and (D + 1) is the homogenous dimensionality

    Example:
        >>> homog_pts = np.array([[1, 2, 3, 4, 5],
        >>>                       [1, 2, 3, 4, 5],
        >>>                       [1, 1, 1, 1, 1]])
        >>> from_homog(homog_pts)
        array([[ 1.,  2.,  3.,  4.,  5.],
               [ 1.,  2.,  3.,  4.,  5.]])
    """
    pts = homog_pts[:-1] / homog_pts[-1][None, :]
    return pts


@profile
def ensure_grayscale(img):
    """
    Checks if an image is grayscale.
    If not it is assumed to be BGR or BGRA and converted.
    """
    n_channels = get_num_channels(img)
    if n_channels == 1:
        img_gray = img
    elif n_channels == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif n_channels == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(
            'input with {} channels is not an image'.format(n_channels))
    return img_gray


@profile
def ensure_float01(img, dtype=np.float32):
    """ Ensure that an image is encoded using a float properly """
    if img.dtype.kind in ('i', 'u'):
        if img.max() > 255:
            raise ValueError('Input image has bad intensity values.')
        img_ = img.astype(dtype) / 255.0
    else:
        img_ = img.astype(dtype)
    return img_


def ensure_uint8(img):
    """ Ensure that an image is encoded in uint8 properly """
    if img.dtype.kind in ('f'):
        if img.max() > 1.0 or img.min() < 0.0:
            raise ValueError('Input image has bad intensity values.')
        img_ = (img * 255.0).astype(np.uint8)
    else:
        img_ = img
    return img_


def get_num_channels(img):
    """ Returns the number of color channels """
    ndims = len(img.shape)
    if ndims == 2:
        nChannels = 1
    elif ndims == 3 and img.shape[2] == 3:
        nChannels = 3
    elif ndims == 3 and img.shape[2] == 4:
        nChannels = 4
    elif ndims == 3 and img.shape[2] == 1:
        nChannels = 1
    else:
        raise ValueError('Cannot determine number of channels '
                         'for img.shape={}'.format(img.shape))
    return nChannels


@profile
def make_channels_comparable(img1, img2):
    """
    Broadcasts image arrays so they can have elementwise operations applied
    """
    if img1.shape != img2.shape:
        c1 = get_num_channels(img1)
        c2 = get_num_channels(img2)
        if len(img1.shape) == 2 and len(img2.shape) == 2:
            raise AssertionError('UNREACHABLE: Both are 2-grayscale')
        elif len(img1.shape) == 3 and len(img2.shape) == 2:
            # Image 2 is grayscale
            if c1 == 3:
                img2 = np.tile(img2[..., None], 3)
            else:
                img2 = img2[..., None]
        elif len(img1.shape) == 2 and len(img2.shape) == 3:
            # Image 1 is grayscale
            if c2 == 3:
                img1 = np.tile(img1[..., None], 3)
            else:
                img1 = img1[..., None]
        elif len(img1.shape) == 3 and len(img2.shape) == 3:
            # Both images have 3 dims.
            # Check if either have color, then check for alpha
            if c1 == 1 and c2 == 1:
                raise AssertionError('UNREACHABLE: Both are 3-grayscale')
            elif c1 == 3 and c2 == 3:
                raise AssertionError('UNREACHABLE: Both are 3-color')
            elif c1 == 1 and c2 == 3:
                img1 = np.tile(img1, 3)
            elif c1 == 3 and c2 == 1:
                img2 = np.tile(img2, 3)
            elif c1 == 3 and c2  == 4:
                # raise NotImplementedError('alpha not handled yet')
                # assumes img1 is in 0:1 format
                img1 = np.dstack((img1, np.ones(img1.shape[0:2])))
            elif c1 == 4 and c2  == 3:
                # assumes img1 is in 0:1 format
                img2 = np.dstack((img2, np.ones(img2.shape[0:2])))
                # raise NotImplementedError('alpha not handled yet')
            else:
                raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
        else:
            raise AssertionError('Unknown shape case: %r, %r' % (img1.shape, img2.shape))
    return img1, img2


@profile
def overlay_alpha_images(img1, img2):
    """
    places img1 on top of img2 respecting alpha channels

    References:
        http://stackoverflow.com/questions/25182421/overlay-numpy-alpha
    """
    img1 = ensure_float01(img1)
    img2 = ensure_float01(img2)

    img1, img2 = make_channels_comparable(img1, img2)

    c1 = get_num_channels(img1)
    c2 = get_num_channels(img2)
    if c1 == 4:
        # alpha1 = np.ascontiguousarray(img1[:, :, 3])
        alpha1 = img1[:, :, 3]
    else:
        alpha1 = np.ones(img1.shape[0:2], dtype=img1.dtype)

    if c2 == 4:
        alpha2 = img2[:, :, 3]
    else:
        alpha2 = np.ones(img2.shape[0:2], dtype=img2.dtype)

    rgb1 = img1[:, :, 0:3]
    rgb2 = img2[:, :, 0:3]

    alpha1_ = alpha1[..., None]
    alpha2_ = alpha2[..., None]
    alpha3_ = alpha1_ + alpha2_ * (1 - alpha1_)

    # rgb3 = rgb1 * alpha1_ + rgb2 * alpha2_

    numer1 = (rgb1 * alpha1_)
    numer2 = (rgb2 * alpha2_ * (1.0 - alpha1_))
    rgb3 = (numer1 + numer2) / alpha3_
    return rgb3


@profile
def make_heatmask(mask, cmap='plasma'):
    """
    Colorizes a single-channel intensity mask (with an alpha channel)
    """
    # import matplotlib as mpl
    # current_backend = mpl.get_backend()
    # for backend in ['Qt5Agg', 'Agg']:
    #     try:
    #         mpl.use(backend, warn=True, force=False)
    #         break
    #     except Exception:
    #         pass
    import matplotlib.pyplot as plt
    assert len(mask.shape) == 2
    mask = ensure_float01(mask)
    heatmask = plt.get_cmap(cmap)(mask)
    heatmask[:, :, 0:3] = heatmask[:, :, 0:3][:, :, ::-1]
    heatmask[:, :, 3] = mask
    return heatmask


@profile
def overlay_heatmask(img, mask, alpha=.9, cmap='plasma'):
    """
    Draws a heatmask on an image using a single-channel intensity mask.
    Colors are computed using a matplotlib colormap.
    """
    heat_mask = make_heatmask(mask, cmap=cmap)
    alpha_mask = heat_mask[:, :, 3].copy()
    # Adjust alpha to be more visible
    alpha_mask[alpha_mask > 0] = alpha  # hack
    heat_mask[:, :, 3] = alpha_mask
    draw_img = overlay_alpha_images(heat_mask, img)
    draw_img = ensure_uint8(draw_img)
    return draw_img
