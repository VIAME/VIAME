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

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess

from vital.types import Image
from vital.types import DetectedObject, DetectedObjectSet
from vital.types import ObjectTrackState, Track, ObjectTrackSet
from vital.types import BoundingBox

import viame.arrows.pytorch.mdnet_tracker as mdnet

# ------------------------------------------------------------------------------
class MDNetTrackerProcess(KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # Config file
        self.add_config_trait("weights_file", "weights_file",
                              'models/mdnet_seed.pth',
                              'MDNet initial weight file for each track.')
        self.declare_config_using_trait("weights_file")

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('initializations', required)
        self.declare_input_port_using_trait('recommendations', optional)
        self.declare_input_port_using_trait('evaluation_requests', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('timestamp', optional)
        self.declare_output_port_using_trait('object_track_set', optional)

        self._track_set = track_set()

    # --------------------------------------------------------------------------
    def _configure(self):
        # Configuration parameters
        self._weights_file = self.config_value('weights_file')

        # Persistent state variables
        self._trackers = dict()
        self._prior_tracks = ObjectTrackState()
        self._base_configure()

    # --------------------------------------------------------------------------
    def _step(self):

        in_img_c = self.grab_input_using_trait('image')
        timestamp = self.grab_input_using_trait('timestamp')
        initializations = self.grab_input_using_trait('initializations')
        recommendations = self.grab_input_using_trait('recommendations')
        requests = self.grab_input_using_trait('evaluation_requests')

        print('mdnet tracker timestamp = {!r}'.format(timestamp))

        # Generate empty output
        output = ObjectTrackSet()

        # Handle new track initialization
        init_tracks = initializations.tracks()
        init_track_ids = [ t.id for t in init_tracks ]
    
        if len( init_tracks ) != 0 or len( self._trackers ) != 0:
            img_npy = 

        for t in init_tracks:
            tid = t.id
            bbox = t.det.box
            self._trackers[tid] = MDNetTracker(img_npy,bbox)

        # Update existing tracks
        for track_id in self._trackers.keys():
            if track_id in init_track_ids:
                continue
            result = self._trackers[ track_id ].update( img_npy )

        # Handle track termination
        # TODO: Remove old or dead tracks

        # Output results
        self.push_to_port_using_trait('timestamp', timestamp)
        self.push_to_port_using_trait('object_track_set', output)

        self._prior_tracks = output
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
