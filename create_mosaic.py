import argparse
import glob
import itertools as itt

import numpy as np
from skimage import (
    io as skio,
    transform as sktr,
)
import tqdm

def read_homog_file(path):
    """Read a homography output file into an Nx3x3 array.
    Coordinate order is Y, X, Z.
    Only reads until the first break.

    """
    with open(path) as f:
        # "start" will contain the starting index
        start = None
        result = []
        for i, line in enumerate(f):
            *matrix, fromf, tof = line.split()
            if start is None:
                assert int(fromf) == int(tof)
                start = int(fromf)
            assert len(matrix) == 9
            assert int(fromf) == i + start
            if int(tof) != start: break
            result.append(list(map(float, matrix)))
    swap_xy = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    return swap_xy @ np.array(result).reshape((-1, 3, 3)) @ swap_xy

def unhomogenize(coords):
    """Signature (i+1, j) -> (i, j)"""
    return coords[..., :-1, :] / coords[..., -1:, :]

def get_extreme_coordinates(homogs, im_size):
    """Return a pair of the UL and BR coordinates"""
    y, x = im_size
    y -= 1; x -= 1
    box = np.array([[0, 0, 1], [y, 0, 1], [0, x, 1], [y, x, 1]]).T
    transformed = unhomogenize(homogs @ box)
    min_yx = np.floor(transformed.min((0, -1))).astype(int)
    max_yx = np.ceil(transformed.max((0, -1))).astype(int)
    return tuple(min_yx), tuple(max_yx)

def translator(offset):
    """Return a matrix that translates homogeneous coordinates by the
    given offset.

    """
    result = np.identity(len(offset) + 1)
    result[:-1, -1] = offset
    return result

def paste(dest, src, src_to_dest):
    """Copy src into dest, transformed as described by src_to_dest, a
    projective matrix that maps a (2D homogeneous) coordinate in src
    to one in dest.

    """
    # Sanity checks
    assert src.ndim == 3 and src.shape[2] in (3, 4)
    if dest.dtype != np.uint8 or src.dtype != np.uint8:
        raise ValueError("Only 8-bit (per channel) images supported")
    y, x = src.shape[:2]
    bbox = np.array([[0, 0, 1], [y, 0, 1], [0, x, 1], [y, x, 1]]).T
    trans_bbox = unhomogenize(src_to_dest @ bbox)
    trans_ul, trans_br = trans_bbox.min(-1), trans_bbox.max(-1)
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

    mask = sktr.warp(
        np.ones(src.shape[:2]), dest_to_src_adj, output_shape=oshape,
    ) > 0.5
    trans = sktr.warp(
        src, dest_to_src_adj, output_shape=oshape,
    )
    # Convert everything back to uint8 (warp converts to double)
    trans = (trans * 255).round().astype(np.uint8)

    dest[tuple(m + ul for m, ul in zip(mask.nonzero(), trans_ul))] = trans[mask]

def paste_many(homogs, ims):
    """Given a sequence of homographies and an iterable of images, produce
    a mosaic image

    """
    ims = iter(ims)
    im = next(ims)
    im_size = im.shape[:2]
    ul, br = get_extreme_coordinates(homogs, im_size)
    # XXX The extra + 1 is a hack
    dest = np.zeros(tuple(np.array(br) - ul + 1 + 1) + (im.shape[2],), dtype=im.dtype)
    for hom, im in zip(homogs, itt.chain([im], ims)):
        assert im.shape[:2] == im_size
        hom = translator(tuple(-x for x in ul)) @ hom
        paste(dest, im, hom)
    return dest

def main(out_file, homog_file, image_glob, frames=None, start=None, stop=None, step=None):
    image_files = sorted(glob.iglob(image_glob))[start:stop]
    homogs = read_homog_file(homog_file)[start:stop]
    length = min(len(image_files), len(homogs))
    if (frames is None) == (step is None):
        raise ValueError("Exactly one of frames and step must be specified")
    if frames is not None:
        frame_numbers = [(length - 1) * i // (frames - 1) for i in range(frames)]
    else:
        frame_numbers = range(0, length, step)
    images = (skio.imread(image_files[i]) for i in tqdm.tqdm(frame_numbers))
    skio.imsave(out_file, paste_many(homogs[frame_numbers], images))

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('out_file', help='Path to output file')
    p.add_argument('homog_file', help='Path to homography file')
    p.add_argument('image_glob', help='(Quoted) glob for input images')
    p.add_argument('--frames', type=int, help='Number of frames represented in output')
    p.add_argument('--start', type=int, metavar='N', help='Ignore first N frames')
    p.add_argument('--stop', type=int, metavar='N', help='Ignore frames after the Nth')
    p.add_argument('--step', type=int, metavar='N', help='Write every Nth frame')
    return p

if __name__ == '__main__':
    main(**vars(create_parser().parse_args()))
