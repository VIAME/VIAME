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
    def center_in_bounds(box, homogs, sizes):
        transform = Homography.matrix_transform
        tc = np.squeeze(transform(homogs, box.center[:, np.newaxis]), -1)
        return ((0 <= tc) & (tc < sizes)).all(-1).any(-1)
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
    homogs_and_sizes = [(np.concatenate([ph, ch]), np.concatenate([ps, cs]))
                        for (ph, ps), (ch, cs)
                        in zip(prev_homogs_and_sizes, curr_homogs_and_sizes)]
    # XXX This could perhaps be vectorized with Numpy
    return [[not center_in_bounds(b, *homogs_and_sizes[c]) for b in boxes]
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
