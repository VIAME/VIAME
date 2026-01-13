# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import logging

import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
)

from .multicam_homog_tracker import MultiHomographyF2F, diff_homogs
from .simple_homog_tracker import (
    Homography, Transformer, add_declare_config,
    get_DetectedObject_bbox, to_DetectedObject_list, wrap_F2FHomography,
)
from .stabilize_many_images import (
    add_declare_input_port, add_declare_output_port,
)

logger = logging.getLogger(__name__)

zero_homog_and_size = (np.empty((0, 3, 3)), np.empty((0, 2), dtype=int))

def get_self_suppression_homogs_and_sizes(multihomog, sizes):
    """Get suppression homographies and sizes within a single timestep

    Returns a value suitable for supplying as arg_suppress_boxes's
    suppression_homogs_and_sizes argument.

    """
    return [
        zero_homog_and_size if cam == hc else (to_curr[s], sizes[s])
        for cam, to_curr in enumerate(diff_homogs(
                multihomog.homogs, multihomog.homogs,
        ))
        for hc in [len(multihomog) // 2]
        # XXX This choice of suppression is very tied to how
        # stabilize_many_images works
        for s in [np.s_[min(cam + 1, hc):max(cam, hc + 1)]]
    ]

def concat_suppression_homogs_and_sizes(*args):
    """Combine multiple values suitable for supplying as
    arg_suppress_boxes's suppression_homogs_and_sizes argument into
    one

    """
    if not args:
        raise ValueError("At least one argument required")
    ncam = len(args[0])
    if any(len(arg) != ncam for arg in args):
        raise ValueError("Arguments must have a consistent number of cameras")
    result = []
    for args_single_cam in zip(*args):
        homogs, sizes = zip(*args_single_cam)
        result.append((np.concatenate(homogs), np.concatenate(sizes)))
    return result

def arg_suppress_boxes(box_lists, suppression_homogs_and_sizes):
    """Compute whether bounding boxes should be kept after suppression

    Arguments:
    - box_lists (list[list[.simple_homog_tracker.BBox]]): bounding
      boxes, where the elements of the outer list correspond to the
      different cameras
    - suppression_homogs_and_sizes (list[tuple[ndarray, ndarray]]):
      transformations to previous frames and those frames' sizes.  The
      elements of the list correspond to the different cameras.  Each
      pair holds:
      - homogs: Nx3x3 ndarray whose first dimension corresponds to
        the previous frames that suppress detections for this camera.
        Each element is a homography that warps camera coordinates to
        previous-frame coordinates.
      - sizes: Nx2 ndarray with the image sizes of the previous frames

    Returns list[list[bool]], False when the corresponding BBox's
    center should have been in a previous frame.

    """
    def center_in_bounds(box, homogs, sizes):
        transform = Homography.matrix_transform
        tc = np.squeeze(transform(homogs, box.center[:, np.newaxis]), -1)
        return ((0 <= tc) & (tc < sizes)).all(-1).any(-1)
    # XXX This could perhaps be vectorized with Numpy
    return [[not center_in_bounds(b, *shs) for b in boxes]
            for boxes, shs in zip(box_lists, suppression_homogs_and_sizes)]

def clip_poly(poly, scores):
    """Clip a poly, only keeping points with a nonnegative score"""
    result = []
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        s1, s2 = scores[i], scores[(i + 1) % len(poly)]
        if s1 == 0:
            result.append(p1)
        else:
            if s1 > 0:
                result.append(p1)
            if s2 != 0 and (s1 > 0) != (s2 > 0):
                result.append((s2 * p1 - s1 * p2) / (s2 - s1))
    return result and np.stack(result)

def transform_poly_to_polys(poly, homog, clip_size):
    """Transform a convex polygon into zero to two clipped polygons

    Polygons are represented as Nx2 array-likes

    """
    # Convert to homogeneous coordinates
    poly = np.concatenate([poly, np.ones((len(poly), 1))], axis=1)
    # Transform (ensuring homog has a non-negative determinant)
    tpoly = poly @ (-homog if np.linalg.det(homog) < 0 else homog).T
    # Cut polygon if needed
    tpolys = [
        clip_poly(tpoly, tpoly[:, 2]),
        clip_poly(-tpoly, -tpoly[:, 2])[::-1],
    ]
    tpolys = [poly for poly in tpolys if len(poly) and (poly[:, 2] != 0).any()]
    if not tpolys:
        raise ValueError("All transformed points lie in camera ground plane")
    # Clip to size
    def clip(poly, scorers):
        for scorer in scorers:
            poly = clip_poly(poly, poly @ scorer)
            if len(poly) == 0:
                break
        return poly
    cpolys = [clip(poly, [
        [1, 0, 0], [0, 1, 0], [-1, 0, clip_size[0]], [0, -1, clip_size[1]],
    ]) for poly in tpolys]
    # Convert from homogeneous coordinates
    return [poly[:, :2] / poly[:, 2:] for poly in cpolys if len(poly) > 2]

def suppression_polys(suppression_homogs_and_sizes, sizes):
    def size_to_poly(size):
        w, h = size
        return [[0, 0], [0, h], [w, h], [w, 0]]
    return [[
        r for h, s in zip(homogs, sizes) for r
        in transform_poly_to_polys(size_to_poly(s), np.linalg.inv(h), size)
    ] for (homogs, sizes), size in zip(suppression_homogs_and_sizes, sizes)]

def wrap_poly(poly, class_):
    result = DetectedObject(
        bbox=BoundingBoxD(*poly.min(0), *poly.max(0)),
        classifications=DetectedObjectType(class_, 1),
    )
    result.set_flattened_polygon(poly.reshape(-1))
    return result

@Transformer.decorate
def find_prev_suppression_homogs_and_sizes():
    """Get suppressing frames from the previous timestep

    Only cameras with the same or an adjacent index are used to
    suppress a given camera.

    The .step call expects parameters two parameters, multihomog and
    sizes, and returns a value suitable for supplying as
    arg_suppress_boxes's suppression_homogs_and_sizes argument.

    """
    prev_multihomog = prev_sizes = None
    output = None
    while True:
        multihomog, sizes = yield output
        if prev_multihomog is None or multihomog.to_id != prev_multihomog.to_id:
            output = len(multihomog) * [zero_homog_and_size]
        else:
            output = [
                (to_prev[s], prev_sizes[s])
                for cam, to_prev in enumerate(diff_homogs(
                        multihomog.homogs, prev_multihomog.homogs,
                ))
                for s in [np.s_[max(0, cam - 1)
                                : min(cam + 2, len(prev_multihomog))]]
            ]
        prev_multihomog, prev_sizes = multihomog, sizes

@Transformer.decorate
def find_all_suppression_homogs_and_sizes():
    frames_by_ref = {}
    output = None
    while True:
        multihomog, sizes = yield output
        try:
            prev_homogs, prev_sizes = frames_by_ref[multihomog.to_id]
        except KeyError:
            prev_homogs, prev_sizes = [], zero_homog_and_size[1]
            output = len(multihomog) * [zero_homog_and_size]
        else:
            output = [(
                to_prev, prev_sizes,
            ) for to_prev in diff_homogs(multihomog.homogs, prev_homogs)]
        prev_homogs.extend(multihomog.homogs)
        prev_sizes = np.concatenate([prev_sizes, sizes])
        frames_by_ref[multihomog.to_id] = prev_homogs, prev_sizes

@Transformer.decorate
def suppress(suppression_poly_class=None, *, past_frames):
    if past_frames == 'prev_neighbors':
        fshs = find_prev_suppression_homogs_and_sizes()
    elif past_frames == 'all':
        fshs = find_all_suppression_homogs_and_sizes()
    else:
        raise ValueError("Invalid value for past_frames")
    output = None
    while True:
        dhss, = yield output
        do_sets, homogs, sizes = zip(*dhss)
        sizes = np.array(sizes)
        multihomog = MultiHomographyF2F.from_homographyf2fs(map(wrap_F2FHomography, homogs))
        do_lists = list(map(to_DetectedObject_list, do_sets))
        boxes = (map(get_DetectedObject_bbox, dos) for dos in do_lists)
        prev_shs = fshs.step(multihomog, sizes)
        curr_shs = get_self_suppression_homogs_and_sizes(multihomog, sizes)
        shs = concat_suppression_homogs_and_sizes(prev_shs, curr_shs)
        keep_its = arg_suppress_boxes(boxes, shs)
        if suppression_poly_class is None:
            poly_dets = [()] * len(do_lists)
        else:
            def n(p):
                # Normalize poly.  This works around DIVE issue #993
                # (https://github.com/Kitware/dive/issues/993)
                assert (p >= 0).all()
                return np.where(p, p, 0)  # Replace -0 with 0
            poly_dets = ((wrap_poly(n(p), suppression_poly_class) for p in ps)
                         for ps in suppression_polys(shs, sizes))
        output = [
            DetectedObjectSet([*(do for k, do in zip(keep, dos) if k), *pd])
            for keep, dos, pd in zip(keep_its, do_lists, poly_dets)
        ]

class MulticamHomogDetSuppressor(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, 'n_input', '2', 'Number of inputs')
        add_declare_config(self, 'suppression_poly_class', '',
                           'If not empty, include polygons indicating the'
                           ' suppressed area with this class')
        add_declare_config(self, 'past_frames', 'prev_neighbors', (
            'Which past frames to use for suppression.  Valid values are'
            ' "prev_neighbors" (previous frame and same and neighboring'
            ' cameras only; this is the default) and "all" (all past frames'
            ' and cameras)'
        ))

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        # XXX work around insufficient wrapping
        self._n_input = int(self.config_value('n_input'))
        for i in range(1, self._n_input + 1):
            add_declare_input_port(self, 'det_objs_' + str(i), 'detected_object_set',
                                   required, 'Input detected object set #' + str(i))
            add_declare_input_port(self, 'homog' + str(i), 'homography_src_to_ref',
                                   required, 'Input homography (source-to-ref) #' + str(i))
            add_declare_input_port(self, 'image' + str(i), 'image', required,
                                   'Input image #' + str(i))
            add_declare_output_port(self, 'det_objs_' + str(i), 'detected_object_set',
                                    optional, 'Output detected object set #' + str(i))

    def _configure(self):
        # XXX actually use this
        self._n_input = int(self.config_value('n_input'))
        spc = self.config_value('suppression_poly_class') or None
        pf = self.config_value('past_frames')
        self._suppressor = suppress(spc, past_frames=pf)
        self._base_configure()

    def _step(self):
        def get_image_size(im):
            return im.width(), im.height()
        dos_list = self._suppressor.step([
            (self.grab_input_using_trait('det_objs_' + str(i)),
             self.grab_input_using_trait('homog' + str(i)),
             get_image_size(self.grab_input_using_trait('image' + str(i))))
            for i in range(1, self._n_input + 1)
        ])
        for i, dos in enumerate(dos_list, 1):
            self.push_to_port_using_trait('det_objs_' + str(i), dos)
        self._base_step()

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.MulticamHomogDetSuppressor'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'multicam_homog_det_suppressor',
        'Multi-camera homography-based detection suppressor',
        MulticamHomogDetSuppressor,
    )
    process_factory.mark_process_module_as_loaded(module_name)
