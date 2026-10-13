# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
import itertools as itt

import numpy as np
import scipy.optimize
from skimage import (
    io as skio,
    transform as sktr,
)
import tqdm

def read_homog_file(path):
    """Read a homography output file into an Nx3x3 array.
    Coordinate order is Y, X, Z.
    Also returns the reference frame for each frame.

    """
    with open(path) as f:
        # "start" will contain the starting index
        start = None
        result = []
        refs = []
        for i, line in enumerate(f):
            *matrix, fromf, tof = line.split()
            if start is None:
                start = int(fromf)
            assert len(matrix) == 9
            assert int(fromf) == i + start
            result.append(list(map(float, matrix)))
            refs.append(int(tof))
    swap_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    result = swap_xy @ np.array(result).reshape((-1, 3, 3)) @ swap_xy
    return result, np.array(refs)

def transform_homog(homog, coords):
    """Signature (n+1, n+1), n -> n"""
    ones = np.ones(coords.shape[:-1] + (1,), dtype=coords.dtype)
    coords = np.concatenate((coords, ones), axis=-1)[..., np.newaxis]
    tcoords = homog @ coords
    tcoords = np.squeeze(tcoords, -1)
    return tcoords[..., :-1] / tcoords[..., -1:]

def get_image_box(im_size):
    y, x = im_size
    y -= 1; x -= 1
    return np.array([[0, 0], [y, 0], [0, x], [y, x]])

def score_homog(homog, im_size):
    """Signature (3, 3) -> () (fixing size as a pair)"""
    box = get_image_box(im_size)
    def get_dists(x):
        """Signature (n, m) -> (n, n)"""
        diffs = np.expand_dims(x, -3) - np.expand_dims(x, -2)
        return (diffs ** 2).sum(-1) ** .5
    tbox = transform_homog(np.expand_dims(homog, -3), box)
    return ((get_dists(tbox) - get_dists(box)) ** 2).sum((-1, -2))

def optimize_homog_fit(homogs, im_size):
    """Return a homography that, when applied after the provided ndarray
    of homographies, minimizes the distortion of images of the given
    size

    """
    def embed(x):
        result = np.empty((3, 3), dtype=x.dtype)
        result.flat[:-1] = x
        result[2, 2] = 1
        return result
    def score(x):
        return score_homog(embed(x) @ homogs, im_size).sum()
    result = scipy.optimize.minimize(
        score, np.eye(3).flat[:-1], method='Nelder-Mead',
    )
    if result.success:
        return embed(result.x)
    else:
        raise RuntimeError(
            "Optimization failed with the message: " + result.message,
        )

def get_extreme_coordinates(homogs, im_size):
    """Return a pair of the UL and BR coordinates"""
    box = get_image_box(im_size)
    if not np.isfinite(homogs).all() or np.any(np.abs(np.linalg.det(homogs)) < 1e-12):
        raise ValueError("Homographies must be finite and nonsingular")
    denominator = (homogs[:, None, 2, :2] * box[None, :, :]).sum(-1) + homogs[:, None, 2, 2]
    if np.any((denominator.min(1) <= 0) & (denominator.max(1) >= 0)):
        raise ValueError("A homography crosses infinity inside the image")
    transformed = transform_homog(homogs[:, np.newaxis], box)
    if not np.isfinite(transformed).all() or np.max(np.abs(transformed)) > 1e12:
        raise ValueError("Homography bounds are not safely representable")
    min_yx = np.floor(transformed.min((0, 1))).astype(int)
    max_yx = np.ceil(transformed.max((0, 1))).astype(int)
    return tuple(min_yx), tuple(max_yx)

def translator(offset):
    """Return a matrix that translates homogeneous coordinates by the
    given offset.

    """
    result = np.identity(len(offset) + 1)
    result[:-1, -1] = offset
    return result

def paste(dest, src, src_to_dest, transparent_black=None):
    """Copy src into dest, transformed as described by src_to_dest, a
    projective matrix that maps a (2D homogeneous) coordinate in src
    to one in dest.

    If transparent_black is true (default false), treat pure black
    pixels as transparent.

    """
    # Sanity checks
    assert src.ndim == 3 and src.shape[2] in (3, 4)
    if dest.dtype != np.uint8 or src.dtype != np.uint8:
        raise ValueError("Only 8-bit (per channel) images supported")
    bbox = get_image_box(src.shape[:2])
    trans_bbox = transform_homog(src_to_dest, bbox)
    trans_ul, trans_br = trans_bbox.min(0), trans_bbox.max(0)
    # Round outward
    trans_ul = np.floor(trans_ul).astype(int)
    trans_br = np.ceil(trans_br).astype(int)
    # Adjust and create
    src_to_dest_adj = translator(-trans_ul) @ src_to_dest
    dest_to_src_adj = np.linalg.inv(src_to_dest_adj)
    # "warp" expects an X,Y coordinate order
    swap_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    dest_to_src_adj = swap_xy @ dest_to_src_adj @ swap_xy
    oshape = tuple(trans_br - trans_ul + 1)

    if transparent_black:
        # TODO: .astype(float) might not be needed
        mask = (src[..., :3] != 0).any(2).astype(float)
    else:
        mask = np.ones(src.shape[:2])
    mask = sktr.warp(mask, dest_to_src_adj, output_shape=oshape) > 0.5
    trans = sktr.warp(
        src, dest_to_src_adj, output_shape=oshape,
    )
    # Convert everything back to uint8 (warp converts to double)
    trans = (trans * 255).round().astype(np.uint8)

    dest_slice = dest[tuple(slice(ul, br + 1) for ul, br in zip(trans_ul, trans_br))]
    np.copyto(dest_slice, trans, where=mask[..., np.newaxis])

def paste_many(homogs, ims, im0, max_pixels=100000000, transparent_black=None):
    """Given a sequence of homographies, an iterable of images, and a
    template image, produce a mosaic image

    """
    im_size = im0.shape[:2]
    ul, br = get_extreme_coordinates(homogs, im_size)
    shape = tuple(int(b) - int(a) + 1 for a, b in zip(ul, br))
    if min(shape) <= 0 or shape[0] * shape[1] > max_pixels:
        raise ValueError(f"Mosaic canvas {shape} exceeds --max-pixels ({max_pixels})")
    dest = np.zeros(shape + (im0.shape[2],), dtype=im0.dtype)
    for hom, im in zip(homogs, ims):
        assert im.shape[:2] == im_size
        hom = translator(tuple(-x for x in ul)) @ hom
        paste(dest, im, hom, transparent_black=bool(transparent_black))
    return dest

def peek_iterable(it):
    it = iter(it)
    x = next(it)
    return x, itt.chain([x], it)

def main(out_file, homogs_and_image_lists, **kwargs):
    if len(homogs_and_image_lists) % 2:
        raise ValueError("Expected an even-length list of"
                         " alternating homographies and image lists")
    hil = iter(homogs_and_image_lists)
    main_multi(out_file, zip(hil, hil), **kwargs)

def main_multi(
        out_file, homogs_and_lists, *, optimize_fit=None, zoom=None,
        frames=None, start=None, stop=None, step=None, reverse=None,
        warp_to_pivot=None, max_pixels=100000000, transparent_black=None,
):
    def read_image_list(path):
        with open(path) as f:
            return [l.rstrip('\n') for l in f]
    images_homogs_refs = ((
        read_image_list(image_list),
        *read_homog_file(homog_file),
    ) for homog_file, image_list in homogs_and_lists)
    images_homogs_refs = list(images_homogs_refs)
    if not images_homogs_refs or any(len(images) != len(homogs) for images, homogs, _ in images_homogs_refs):
        raise ValueError("Each image list must match its homography count")
    if zoom is not None and (not np.isfinite(zoom) or zoom <= 0):
        raise ValueError("Zoom must be finite and positive")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    images_homogs_refs = [tuple(
        x[start:stop] for x in ihr
    ) for ihr in images_homogs_refs]
    length = min(len(x) for ihr in images_homogs_refs for x in ihr)
    if length == 0:
        raise ValueError("No frames selected")
    if frames is not None and frames < 1 or step is not None and step < 1:
        raise ValueError("frames and step must be positive")
    if (frames is None) == (step is None):
        raise ValueError("Exactly one of frames and step must be specified")
    if frames is not None:
        frame_numbers = [0] if frames == 1 else [(length - 1) * i // (frames - 1) for i in range(frames)]
    else:
        frame_numbers = range(0, length, step)
    if reverse:
        frame_numbers = frame_numbers[::-1]
        images_homogs_refs = images_homogs_refs[::-1]
    if len({
            x for _, _, refs in images_homogs_refs
            for x in np.unique(refs[frame_numbers])
    }) > 1:
        raise ValueError("Requested frames do not all have the same reference")
    images = (skio.imread(image_files[i])
              for i in tqdm.tqdm(frame_numbers)
              for image_files, _, _ in images_homogs_refs)
    im0, images = peek_iterable(images)
    rel_homogs = np.stack([
        homogs[frame_numbers] for _, homogs, _ in images_homogs_refs
    ], axis=1).reshape((-1, 3, 3))
    if warp_to_pivot:
        pivot = len(rel_homogs)-1
        inv = np.linalg.inv(rel_homogs[pivot])
        rel_homogs[:] = [np.matmul(inv, a) for a in rel_homogs]
    if optimize_fit:
        rel_homog_0 = rel_homogs[-1 if reverse else 0]
        rel_homogs = np.linalg.inv(rel_homog_0) @ rel_homogs
        fit_homog = optimize_homog_fit(rel_homogs, im0.shape[:2])
        rel_homogs = fit_homog @ rel_homogs
    if zoom is not None:
        rel_homogs = np.diag([zoom, zoom, 1]) @ rel_homogs
    skio.imsave(out_file, paste_many(
        rel_homogs, images, im0, max_pixels=max_pixels,
        transparent_black=bool(transparent_black),
    ))
    () = images  # Finally move the progress bar to 100%

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('out_file', help='Path to output file')
    p.add_argument('homogs_and_image_lists', nargs='+', metavar='homog/image_list',
                   help='Even-length list of alternating paths to'
                   ' homography files and paths to files with'
                   ' newline-separated image paths, one pair per camera')
    p.add_argument('--max-pixels', type=int, default=100000000, help='Maximum canvas pixels to allocate')
    p.add_argument('--frames', type=int, help='Number of frames represented in output')
    p.add_argument('--optimize-fit', action='store_true', help='Apply an additional transformation to all images to minimize distortion')
    p.add_argument('--reverse', action='store_true', help='Render images in reverse order')
    p.add_argument('--start', type=int, metavar='N', help='Ignore first N frames')
    p.add_argument('--stop', type=int, metavar='N', help='Ignore frames after the Nth')
    p.add_argument('--step', type=int, metavar='N', help='Write every Nth frame')
    p.add_argument('--transparent-black', action='store_true', help='Treat black in input images as transparent')
    p.add_argument('--zoom', type=float, metavar='Z', help='Scale the output image by a factor of Z')
    return p

if __name__ == '__main__':
    main(**vars(create_parser().parse_args()))
