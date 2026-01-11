# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

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

from .utilities import Grid, gpu_list_desc, parse_gpu_list

class ResnetDescriptors(KwiverProcess):

    # --------------------------------------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # GPU list
        self.add_config_trait("gpu_list", "gpu_list", 'all',
                              gpu_list_desc(use_for='Resnet descriptors'))
        self.declare_config_using_trait('gpu_list')

        # Resnet
        #-----------------------------------------------------------------------
        self.add_config_trait("model_arch", "model_arch", 'resnet',
                              'Model architecture (res, alex, or efficientnet)')
        self.declare_config_using_trait('model_arch')

        self.add_config_trait("model_path", "model_path",
                              'models/resnet50_default.pt',
                              'Trained PyTorch model.')
        self.declare_config_using_trait('model_path')

        self.add_config_trait("model_input_size", "model_input_size", '224',
                              'Model input image size')
        self.declare_config_using_trait('model_input_size')

        self.add_config_trait("batch_size", "batch_size", '128',
                              'resnet model processing batch size')
        self.declare_config_using_trait('batch_size')
        #-----------------------------------------------------------------------

        # detection select threshold
        self.add_config_trait("detection_select_threshold",
                              "detection_select_threshold",
                              '0.0',
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
        self.declare_input_port_using_trait('timestamp', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('detected_object_set', optional)

    # --------------------------------------------------------------------------
    def _configure(self):
        self._select_threshold = float(self.config_value('detection_select_threshold'))

        # gpu_list
        self._gpu_list = parse_gpu_list(self.config_value('gpu_list'))

        # Resnet model config
        model_arch = self.config_value('model_arch')
        model_path = self.config_value('model_path')
        img_size = int(self.config_value('model_input_size'))
        batch_size = int(self.config_value('batch_size'))

        if model_arch == "resnet":
            from .torchvision.resnet_feature_extractor import ResnetFeatureExtractor
            self._app_feature_extractor = ResnetFeatureExtractor(model_path,
                img_size, batch_size, self._gpu_list)
        elif model_arch == "alexnet":
            from .torchvision.alexnet_feature_extractor import AlexNetFeatureExtractor
            self._app_feature_extractor = AlexnetFeatureExtractor(model_path,
                img_size, batch_size, self._gpu_list)
        elif model_arch == "efficientnet":
            from .torchvision.enet_feature_extractor import EfficientNetFeatureExtractor
            self._app_feature_extractor = EfficientNetFeatureExtractor(model_path,
                img_size, batch_size, self._gpu_list)
        else:
            print( "Invalid model architecture: " + model_arch )
            return False

        # Init variables
        self._grid = Grid()
        self._base_configure()

    # --------------------------------------------------------------------------
    def _step(self):
        try:
            # Grab image container from port using traits
            in_img_c = self.grab_input_using_trait('image')
            dos_ptr = self.grab_input_using_trait('detected_object_set')
            if self.has_input_port_edge_using_trait('timestamp'):
                timestamp = self.grab_input_using_trait('timestamp')
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
                    bbox = item.bounding_box
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
        print('!!!!Resnet tracking Deleting python process!!!!')

# ==============================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch.pytorch_descriptors'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('pytorch_descriptors',
                                'pytorch feature extraction',
                                ResnetDescriptors)

    process_factory.mark_process_module_as_loaded(module_name)
