# ckwg +29
# Copyright 2019-2020 by Kitware, Inc.
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

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import numpy as np

from kwiver.kwiver_process import KwiverProcess
from sprokit.pipeline import process

from vital.types import Image
from vital.types import BoundingBox
from vital.types import DetectedObject, DetectedObjectSet
from vital.types import ObjectTrackState, Track, ObjectTrackSet

import viame.arrows.pytorch.mdnet_tracker as mdnet

# ------------------------------------------------------------------------------
class MDNetTrackerProcess(KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # Config file
        self.add_config_trait("weights_file", "weights_file",
          'models/mdnet_seed.pth',
          'MDNet initial weight file for each object track.')
        self.declare_config_using_trait("weights_file")
        self.add_config_trait("init_method", "init_method",
          'external_only',
          'Method for initializing new tracks, can be: external_only or '
          'using_detections.')
        self.declare_config_using_trait("init_method")
        self.add_config_trait("init_threshold", "init_threshold",
          '0.20',
          'If tracking multiple targets, the initialization threshold over '
          'detected object classifications for new tracks')
        self.declare_config_using_trait("init_threshold")
        self.add_config_trait("iou_threshold", "iou_threshold",
          '0.50',
          'If tracking multiple targets, the initialization threshold over '
          'box intersections in order to generate new tracks')
        self.declare_config_using_trait("iou_threshold")
        self.add_config_trait("type_string", "type_string",
          '',
          'If non-empty set the output track to be a track of this '
          'object category.')
        self.declare_config_using_trait("type_string")

        # add non-standard input and output elements
        self.add_port_trait( "initializations",
          "object_track_set", "Input external track initializations" )
        self.add_port_trait( "recommendations",
          "object_track_set", "Input external track recommendations" )
        self.add_port_trait( "evaluation_requests",
          "detected_object_set", "External detections to test if target" )
        self.add_port_trait( "evaluations",
          "detected_object_set", "Completed evaluations with scores" )

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('detected_object_set', optional)
        self.declare_input_port_using_trait('initializations', required)
        self.declare_input_port_using_trait('recommendations', optional)
        self.declare_input_port_using_trait('evaluation_requests', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('timestamp', optional)
        self.declare_output_port_using_trait('object_track_set', optional)
        self.declare_output_port_using_trait('evaluations', optional)

    # --------------------------------------------------------------------------
    def _configure(self):
        # Configuration parameters
        self._weights_file = str(self.config_value('weights_file'))
        self._init_threshold = str(self.config_value('init_method'))
        self._init_threshold = float(self.config_value('init_threshold'))
        self._iou_threshold = float(self.config_value('iou_threshold'))
        self._type_string = str(self.config_value('type_string'))

        # Load model only once across all tracks for speed
        mdnet.opts['model_path'] = mdnet.MDNet(self._weights_file)

        # Persistent state variables
        self._trackers = dict()
        self._tracks = dict()
        self._base_configure()

    # --------------------------------------------------------------------------
    def _step(self):

        # Get all inputs even ones we don't use
        in_img_c = self.grab_input_using_trait('image')
        timestamp = self.grab_input_using_trait('timestamp')
        detections = self.grab_input_using_trait('detected_object_set')
        initializations = self.grab_input_using_trait('initializations')
        recommendations = self.grab_input_using_trait('recommendations')
        requests = self.grab_input_using_trait('evaluation_requests')

        print('mdnet tracker timestamp = {!r}'.format(timestamp))

        # Handle new track external initialization
        init_tracks = initializations.tracks()
        init_track_ids = [t.id() for t in init_tracks]
    
        if len(init_tracks) != 0 or len(self._trackers) != 0:
            img_npy = in_img_c.image().asarray().astype('uint8')

        for t in init_tracks:
            if t.last_frame().timestamp() is not timestamp:
                continue
            tid = t.id()
            cbox = t.last_frame().detection().bounding_box()
            bbox = [cbox.min_x(), cbox.min_y(), cbox.width(), cbox.height()]
            self._trackers[tid] = MDNetTracker(img_npy, bbox)
            self._tracks[tid] = t

        # Update existing tracks
        for tid in self._trackers.keys():
            if tid in init_track_ids:
                continue # Already processed (initialized) on frame
            bbox, score = self._trackers[tid].update(img_npy)
            if score > opts['link_score_required']:
                bbox = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
                cbox = BoundingBox(bbox)
                self._tracks[tid].append(ObjectTrackState(timestamp, cbox))

        # Handle track termination
        # TODO: Remove old or dead tracks

        # Classify requested evaluations
        # TODO: Evaluate input detections
        output_evaluations = DetectedObjectSet()

        # Output results
        output_tracks = ObjectTrackSet(
          [Track(tid, trks) for tid, trk in self._tracks.items()])

        self.push_to_port_using_trait('timestamp', timestamp)
        self.push_to_port_using_trait('object_track_set', output_tracks)
        self.push_to_port_using_trait('evaluations', output_evaluations)
        self._base_step()

# ==============================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:viame.pytorch.MDNetTrackerProcess'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('mdnet_tracker',
      'Object tracking using mdnet', MDNetTrackerProcess)

    process_factory.mark_process_module_as_loaded(module_name)
