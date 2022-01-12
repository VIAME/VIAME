# ckwg +29
# Copyright 2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

def get_suppression_homogs_and_sizes(
        multihomog, sizes, prev_multihomog, prev_sizes):
    """Compute for each camera transformations to other frames that
    suppress its detections and their sizes

    Returns a list of pairs, one per camera, where the first element
    is an ndarray of homographies from the camera to suppressing
    frames and the second is an ndarray of those frames' sizes.

    """
    zero_homog_and_size = (np.empty((0, 3, 3)), np.empty((0, 2), dtype=int))
    if prev_multihomog is None or multihomog.to_id != prev_multihomog.to_id:
        prev_homogs_and_sizes = len(multihomog) * [zero_homog_and_size]
    else:
        prev_homogs_and_sizes = [
            (to_prev[s], prev_sizes[s])
            for cam, to_prev in enumerate(diff_homogs(
                    multihomog.homogs, prev_multihomog.homogs,
            ))
            for s in [np.s_[max(0, cam - 1):min(cam + 2, len(prev_multihomog))]]
        ]
    curr_homogs_and_sizes = [
        zero_homog_and_size if cam == hc else (to_curr[s], sizes[s])
        for cam, to_curr in enumerate(diff_homogs(
                multihomog.homogs, multihomog.homogs,
        ))
        for hc in [len(multihomog) // 2]
        # XXX This choice of suppression is very tied to how
        # stabilize_many_images works
        for s in [np.s_[min(cam + 1, hc):max(cam, hc + 1)]]
    ]
    return [(np.concatenate([ph, ch]), np.concatenate([ps, cs]))
            for (ph, ps), (ch, cs)
            in zip(prev_homogs_and_sizes, curr_homogs_and_sizes)]

def arg_suppress_boxes(box_lists, suppression_homogs_and_sizes):
    """Return a list of iterables of bools, False when the corresponding
    BBox should have been in the previous frame.

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
def suppress(suppression_poly_class=None):
    prev_multihomog = prev_sizes = None
    output = None
    while True:
        dhss, = yield output
        do_sets, homogs, sizes = zip(*dhss)
        sizes = np.array(sizes)
        multihomog = MultiHomographyF2F.from_homographyf2fs(map(wrap_F2FHomography, homogs))
        do_lists = list(map(to_DetectedObject_list, do_sets))
        boxes = (map(get_DetectedObject_bbox, dos) for dos in do_lists)
        shs = get_suppression_homogs_and_sizes(multihomog, sizes,
                                               prev_multihomog, prev_sizes)
        keep_its = arg_suppress_boxes(boxes, shs)
        if suppression_poly_class is None:
            poly_dets = [()] * len(do_lists)
        else:
            poly_dets = ((wrap_poly(p, suppression_poly_class) for p in ps)
                         for ps in suppression_polys(shs, sizes))
        prev_multihomog, prev_sizes = multihomog, sizes
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
        self._suppressor = suppress(spc)
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
    module_name = 'python:kwiver.python.MulticamHomogDetSuppressor'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'multicam_homog_det_suppressor',
        'Multi-camera homography-based detection suppressor',
        MulticamHomogDetSuppressor,
    )
    process_factory.mark_process_module_as_loaded(module_name)
