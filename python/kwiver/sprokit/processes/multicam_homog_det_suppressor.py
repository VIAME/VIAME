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
from kwiver.vital.types import DetectedObjectSet

from .multicam_homog_tracker import MultiHomographyF2F, diff_homogs
from .simple_homog_tracker import (
    Homography, Transformer, add_declare_config,
    get_DetectedObject_bbox, to_DetectedObject_list, wrap_F2FHomography,
)
from .stabilize_many_images import (
    add_declare_input_port, add_declare_output_port,
)

logger = logging.getLogger(__name__)

def arg_suppress_boxes(
        box_lists, multihomog, sizes, prev_multihomog, prev_sizes):
    """Return a list of iterables of bools, False when the corresponding
    BBox should have been in the previous frame.

    """
    def in_bounds(centers, sizes):
        """Signature (n, m), (n, m) -> ()"""
        return ((0 <= centers) & (centers < sizes)).all(-1).any(-1)
    def trans_center(homog, box):
        transform = Homography.matrix_transform
        return np.squeeze(transform(homog, box.center[:, np.newaxis]), -1)
    if prev_multihomog is None or multihomog.to_id != prev_multihomog.to_id:
        def in_prev(cam, box):
            return False
    else:
        curr_to_prev = diff_homogs(multihomog.homogs, prev_multihomog.homogs)
        # XXX This could perhaps be vectorized with Numpy
        def in_prev(cam, box):
            s = np.s_[max(0, cam - 1):min(cam + 2, len(prev_multihomog))]
            # XXX This could handle behind-camera points specially.
            return in_bounds(trans_center(curr_to_prev[cam, s], box),
                             prev_sizes[s])
    curr_to_curr = diff_homogs(multihomog.homogs, multihomog.homogs)
    def in_curr(cam, box):
        hc = len(multihomog) // 2
        if cam == hc:
            return False
        # XXX This choice of suppression is very tied to how
        # stabilize_many_images works
        s = np.s_[min(cam + 1, hc):max(cam, hc + 1)]
        return in_bounds(trans_center(curr_to_curr[cam, s], box), sizes[s])
    return [[not (in_prev(c, b) or in_curr(c, b)) for b in boxes]
            for c, boxes in enumerate(box_lists)]

@Transformer.decorate
def suppress():
    prev_multihomog = prev_sizes = None
    output = None
    while True:
        dhss, = yield output
        do_sets, homogs, sizes = zip(*dhss)
        sizes = np.array(sizes)
        multihomog = MultiHomographyF2F.from_homographyf2fs(map(wrap_F2FHomography, homogs))
        do_lists = list(map(to_DetectedObject_list, do_sets))
        boxes = (map(get_DetectedObject_bbox, dos) for dos in do_lists)
        keep_its = arg_suppress_boxes(boxes, multihomog, sizes,
                                      prev_multihomog, prev_sizes)
        prev_multihomog, prev_sizes = multihomog, sizes
        output = [DetectedObjectSet([do for k, do in zip(keep, dos) if k])
                  for keep, dos in zip(keep_its, do_lists)]

class MulticamHomogDetSuppressor(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, 'n_input', '2', 'Number of inputs')

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
        self._suppressor = suppress()
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
