# -*- coding: utf-8 -*-
"""
MOSTLY DEPRECATE FOR KWIMAGE
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import glob
from os.path import expanduser, exists, join, basename
import numpy as np


def load_image_paths(dpath, ext=('.png', '.tiff', 'tif')):
    # DEPRECATE
    dpath = expanduser(dpath)
    if not exists(dpath):
        raise ValueError('dpath = {} does not exist'.format(dpath))
    if not isinstance(ext, (list, tuple)):
        ext = [ext]

    image_paths = []
    for ext_ in ext:
        image_paths.extend(list(glob.glob(join(dpath, '*' + ext_))))
    # potentially non-general
    # (utilfname solves this though)
    image_paths = sorted(image_paths, key=basename)
    return image_paths


def image_slices(img_shape, target_shape, overlap=0, keepbound=False):
    """
    Generates "sliding window" slices to break a large image into smaller
    pieces.

    Args:
        img_shape (tuple): height and width of the image

        target_shape (tuple): (height, width) of the

        overlap (float): a number between 0 and 1 indicating the fraction of
            overlap that parts will have. Must be `0 <= overlap < 1`.

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

    Yields:
        tuple(slice, slice): row and column slices used for numpy indexing

    Example:
        >>> img_shape = (2000, 2000)
        >>> target_shape = (360, 480)
        >>> overlap = 0
        >>> keepbound = True
        >>> list(image_slices(img_shape, target_shape, overlap, keepbound))
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError(('part overlap was {}, but it must be '
                          'in the range [0, 1)').format(overlap))
    ph, pw = target_shape
    sy = int(ph - ph * overlap)
    sx = int(pw - pw * overlap)
    orig_h, orig_w = img_shape
    kw = dict(keepbound=keepbound, check=False, start=0)
    for rslice in wide_strides_1d(ph, orig_h, sy, **kw):
        for cslice in wide_strides_1d(pw, orig_w, sx, **kw):
            yield rslice, cslice


def overlay_colorized(colorized, orig, alpha=.6, keepcolors=False):
    """
    Overlays a color segmentation mask on an original image

    Args:
        colorized (ndarray): the color mask to be overlayed on top of the original image
        orig (ndarray): the original image to superimpose on
        alpha (float): blend level to use if colorized is not an alpha image

    """
    import kwimage
    color_mask = kwimage.ensure_alpha_channel(colorized, alpha=alpha)
    if not keepcolors:
        orig = ensure_grayscale(orig)
    color_blend = kwimage.overlay_alpha_images(color_mask, orig)
    color_blend = (color_blend * 255).astype(np.uint8)
    return color_blend


def wide_strides_1d(margin, stop, step=None, start=0, keepbound=False,
                    check=True):
    """
    Helper for `image_slices`. Generates slices in a single dimension.

    Args:
        start (int): starting point (in most cases set this to 0)

        margin (int): the length of the slice (window)

        stop (int): the length of the image dimension

        step (int): the length of each step / distance between slices

        keepbound (bool): if True, a non-uniform step will be taken to ensure
            that the right / bottom of the image is returned as a slice if
            needed. Such a slice will not obey the overlap constraints.
            (Defaults to False)

        check (bool): if True an error will be raised if the window does not
            cover the entire extent from start to stop, even if keepbound is
            True.

    Yields:
        slice : slice in one dimension of size (margin)

    Example:
        >>> stop, margin, step = 2000, 360, 360
        >>> keepbound = True
        >>> strides = list(wide_strides_1d(margin, stop, step, keepbound, check=False))
        >>> assert all([(s.stop - s.start) == margin for s in strides])

    Example:
        >>> stop, margin, step = 200, 46, 7
        >>> keepbound = True
        >>> strides = list(wide_strides_1d(margin, stop, step, keepbound=False, check=True))
        >>> starts = np.array([s.start for s in strides])
        >>> stops = np.array([s.stop for s in strides])
        >>> widths = stops - starts
        >>> assert np.all(np.diff(starts) == step)
        >>> assert np.all(widths == margin)

    Example:
        >>> import pytest
        >>> stop, margin, step = 200, 36, 7
        >>> with pytest.raises(ValueError):
        ...     list(wide_strides_1d(margin, stop, step))
    """
    if step is None:
        step = margin

    if check:
        # see how far off the end we would fall if we didnt check bounds
        perfect_final_pos = (stop - start - margin)
        overshoot = perfect_final_pos % step
        if overshoot > 0:
            raise ValueError(
                ('margin={} and step={} overshoot endpoint={} '
                 'by {} units when starting from={}').format(
                     margin, step, stop, overshoot, start))
    pos = start
    # probably could be more efficient with numpy here
    while True:
        endpos = pos + margin
        yield slice(pos, endpos)
        # Stop once we reached the end
        if endpos == stop:
            break
        pos += step
        if pos + margin > stop:
            if keepbound:
                # Ensure the boundary is always used even if steps
                # would overshoot Could do some other strategy here
                pos = stop - margin
            else:
                break


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


def ensure_grayscale(img, colorspace_hint='BGR'):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> import numpy as np
        >>> ensure_grayscale(np.array([[[0, 0, 1]]], dtype=np.float32))
        array([[0.299]], dtype=float32)
    """
    import kwimage
    img = kwimage.ensure_float01(img, copy=False)
    c = get_num_channels(img)
    if c == 1:
        return img
    else:
        return kwimage.convert_colorspace(img, src_space=colorspace_hint,
                                          dst_space='gray')


def adjust_gamma(img, gamma=1.0):
    """
    gamma correction function

    References:
        http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/

    Ignore:
        >>> # xdoctest: +REQUIRES(module:kwplot, module:kwimage)
        >>> import kwimage
        >>> fpath = kwimage.grab_test_image()
        >>> img = kwimage.imread(fpath)
        >>> gamma = .5
        >>> imgf = ensure_float01(img)
        >>> img2 = adjust_gamma(img, gamma)
        >>> img3 = adjust_gamma(imgf, gamma)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img, pnum=(3, 3, 1), fnum=1)
        >>> kwplot.imshow(img2, pnum=(3, 3, 2), fnum=1)
        >>> kwplot.imshow(img3, pnum=(3, 3, 3), fnum=1)
        >>> kwplot.imshow(adjust_gamma(img, 1), pnum=(3, 3, 5), fnum=1)
        >>> kwplot.imshow(adjust_gamma(imgf, 1), pnum=(3, 3, 6), fnum=1)
        >>> kwplot.imshow(adjust_gamma(img, 2), pnum=(3, 3, 8), fnum=1)
        >>> kwplot.imshow(adjust_gamma(imgf, 2), pnum=(3, 3, 9), fnum=1)
    """
    import cv2
    if img.dtype.kind in ('i', 'u'):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        table = (((np.arange(0, 256) / 255.0) ** (1 / gamma)) * 255).astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)
        ]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)
    else:
        import kwimage
        np_img = kwimage.ensure_float01(img, copy=False)
        gain = 1
        np_img = gain * (np_img ** (1 / gamma))
        np_img = np.clip(np_img, 0, 1)
        return np_img


def _lookup_cv2_colorspace_conversion_code(src_space, dst_space):
    import cv2
    src = src_space.upper()
    dst = dst_space.upper()
    convert_attr = 'COLOR_{}2{}'.format(src, dst)
    if not hasattr(cv2, convert_attr):
        prefix = 'COLOR_{}2'.format(src)
        valid_dst_spaces = [
            key.replace(prefix, '')
            for key in cv2.__dict__.keys() if key.startswith(prefix)]
        raise KeyError(
            '{} does not exist, valid conversions from {} are to {}'.format(
                convert_attr, src_space, valid_dst_spaces))
    else:
        code = getattr(cv2, convert_attr)
    return code


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.util.imutil all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
