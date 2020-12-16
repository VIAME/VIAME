#ckwg +28
# Copyright 2019 by Kitware, Inc.
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
import threading

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.vital.types import DetectedObject, DetectedObjectSet
from kwiver.vital.types import new_descriptor

from timeit import default_timer as timer

from kwiver.vital.util.VitalPIL import get_pil_image

from .utils.alexnet_feature_extractor import AlexNetFeatureExtractor
from .utils.parse_gpu_list import gpu_list_desc, parse_gpu_list

class AlexNetDescriptors(KwiverProcess):

    # -------------------------------------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # GPU list
        self.add_config_trait("gpu_list", "gpu_list", 'all',
                              gpu_list_desc(use_for='alexnet descriptors'))
        self.declare_config_using_trait('gpu_list')

        # alexnet
        #----------------------------------------------------------------------------------
        self.add_config_trait("alexnet_model_path", "alexnet_model_path",
                              'models/alexnet_default.pt',
                              'Trained PyTorch model.')
        self.declare_config_using_trait('alexnet_model_path')

        self.add_config_trait("alexnet_model_input_size", "alexnet_model_input_size", '224',
                              'Model input image size')
        self.declare_config_using_trait('alexnet_model_input_size')

        self.add_config_trait("alexnet_batch_size", "alexnet_batch_size", '128',
                              'alexnet model processing batch size')
        self.declare_config_using_trait('alexnet_batch_size')
        #----------------------------------------------------------------------------------

        # detection select threshold
        self.add_config_trait("detection_select_threshold", "detection_select_threshold", '0.0',
                              'detection select threshold')
        self.declare_config_using_trait('detection_select_threshold')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        # self.declare_input_port_using_trait('framestamp', optional)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('timestamp', required)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('detected_object_set', optional)

    # ----------------------------------------------
    def _configure(self):
        self._select_threshold = float(self.config_value('detection_select_threshold'))

        # GPU_list
        self._GPU_list = parse_gpu_list(self.config_value('gpu_list'))

        # alexnet model config
        alexnet_img_size = int(self.config_value('alexnet_model_input_size'))
        alexnet_batch_size = int(self.config_value('alexnet_batch_size'))
        alexnet_model_path = self.config_value('alexnet_model_path')
        self._app_feature_extractor = AlexNetFeatureExtractor(alexnet_model_path, alexnet_img_size, alexnet_batch_size, self._GPU_list)

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        try:
            # Grab image container from port using traits
            in_img_c = self.grab_input_using_trait('image')
            timestamp = self.grab_input_using_trait('timestamp')
            dos_ptr = self.grab_input_using_trait('detected_object_set')
            print('timestamp = {!r}'.format(timestamp))

            # Get current frame and give it to app feature extractor
            im = get_pil_image(in_img_c.image())
            self._app_feature_extractor.frame = im

            bbox_num = 0

            # Get detection bbox
            dos = dos_ptr.select(self._select_threshold)
            bbox_num = dos.size()

            det_obj_set = DetectedObjectSet()

            if bbox_num == 0:
                print('!!! No bbox is provided on this frame and skip this frame !!!')
            else:
                # appearance features (format: pytorch tensor)
                app_f_begin = timer()
                pt_app_features = self._app_feature_extractor(dos, False)
                app_f_end = timer()
                print('%%%app feature eclapsed time: {}'.format(app_f_end - app_f_begin))

                # get new track state from new frame and detections
                for idx, item in enumerate(dos):
                    bbox = item.bounding_box()
                    fid = timestamp.get_frame()
                    ts = timestamp.get_time_usec()
                    d_obj = item

                    # store app feature to detectedObject
                    app_f = new_descriptor(pt_app_features[idx].numpy().size)
                    app_f[:] = pt_app_features[idx].numpy()
                    # print( pt_app_features[idx].numpy() )
                    d_obj.set_descriptor(app_f)
                    det_obj_set.add(d_obj)

            # push track set to output port
            self.push_to_port_using_trait('detected_object_set', det_obj_set)
            self._base_step()

        except BaseException as e:
            print( repr( e ) )
            import traceback
            print( traceback.format_exc() )
            sys.stdout.flush()
            raise

    def __del__(self):
        print('!!!!alexnet tracking Deleting python process!!!!')

# ==================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.alexnet_descriptors'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('alexnet_descriptors', 'alexnet feature extraction',
                                AlexNetDescriptors)

    process_factory.mark_process_module_as_loaded(module_name)
