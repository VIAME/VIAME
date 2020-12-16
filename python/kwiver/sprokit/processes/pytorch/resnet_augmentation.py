# ckwg +28
# Copyright 2018 by Kitware, Inc.
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
import torch
import pickle

from torchvision import models, transforms
from torch.autograd import Variable
from torch import nn

import numpy as np
import scipy as sp
import scipy.optimize
import threading

from PIL import Image as pilImage

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.pipeline import datum
from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.vital.types import Image
from kwiver.vital.types import DetectedObject, DetectedObjectSet
from kwiver.vital.types import new_descriptor, DescriptorSet

from timeit import default_timer as timer

from kwiver.vital.util.VitalPIL import get_pil_image

from .utils.grid import Grid
from .utils.augmented_resnet_feature_extractor import AugmentedResnetFeatureExtractor
from .utils.parse_gpu_list import gpu_list_desc, parse_gpu_list

def to_vital(raw_data):
    if len(raw_data) == 0:
        return DescriptorSet()
    vital_descriptors = []
    for item in raw_data:
        new_desc=new_descriptor(item.size)
        new_desc[:]=item
        vital_descriptors.append(new_desc)
    return DescriptorSet(vital_descriptors)

class DataAugmentation(KwiverProcess):

    # -------------------------------------------------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # GPU list
        #----------------------------------------------------------------------------------
        self.add_config_trait("gpu_list",
                              "gpu_list",
                              'all',
                              gpu_list_desc(use_for='augmentation'))
        self.declare_config_using_trait('gpu_list')

        # Resnet 
        #----------------------------------------------------------------------------------
        self.add_config_trait("resnet_model_path",
                              "resnet_model_path",
                              'resnet/snapshot_epoch_6.pt',
                              'Trained PyTorch model.')
        self.declare_config_using_trait('resnet_model_path')

        self.add_config_trait("resnet_model_input_size",
                              "resnet_model_input_size",
                              '224',
                              'Model input image size')
        self.declare_config_using_trait('resnet_model_input_size')

        self.add_config_trait("resnet_batch_size",
                              "resnet_batch_size",
                              '2',
                              'resnet model processing batch size')
        self.declare_config_using_trait('resnet_batch_size')
        #----------------------------------------------------------------------------------

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

        #----------------------------------------------------------------------------------
        self.add_config_trait("use_historical_database",
                              "use_historical_database",
                              'false',
                              'Historical database for negative exemplars')
        self.declare_config_using_trait('use_historical_database')
        self.add_config_trait("historical_tree",
                              "historical_tree",
                              'historical_tree.kdtree.pickle',
                              'Historical database for negative exemplars')
        self.declare_config_using_trait('historical_tree')
        self.add_config_trait("historical_database",
                              "historical_database",
                              'historical_descriptors.kdtree.pickle',
                              'Historical database for negative exemplars')
        self.declare_config_using_trait('historical_database')
        self.add_config_trait("negative_sample_count",
                              "negative_sample_count",
                              '1000',
                              'Number of negative samples to use from the database')
        self.declare_config_using_trait('negative_sample_count')
        self.add_config_trait("rotational_shifts",
                              "rotational_shifts",
                              '36',
                              'Augmentation rotational shifts')
        self.declare_config_using_trait('rotational_shifts')
        self.add_config_trait("resize_factor",
                              "resize_factor",
                              '0.2',
                              'Augmentation scale shift factor')
        self.declare_config_using_trait('resize_factor')
        self.add_config_trait("int_shift_factor",
                              "int_shift_factor",
                              '0.2',
                              'Augmentation intensity shift factor')
        self.declare_config_using_trait('int_shift_factor')


        # Custom port IDs
        #----------------------------------------------------------------------------------
        self.add_port_trait("new_positive_descriptors", "descriptor_set",
                            "Positive exemplar descriptor set")
        self.add_port_trait("new_positive_ids", "string_vector",
                            "Positive exemplar descriptor UUIDs")
        self.add_port_trait("new_negative_descriptors", "descriptor_set",
                            "Negative exemplar descriptor set")
        self.add_port_trait("new_negative_ids", "string_vector",
                            "Negative exemplar descriptor UUIDs")

        #  input port ( port-name,flags)
        # self.declare_input_port_using_trait('framestamp', optional)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('detected_object_set', required)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('new_positive_descriptors', optional)
        self.declare_output_port_using_trait('new_positive_ids', optional)
        self.declare_output_port_using_trait('new_negative_descriptors', optional)
        self.declare_output_port_using_trait('new_negative_ids', optional)

    # -------------------------------------------------------------------------------------
    def _configure(self):
        self._select_threshold = float(self.config_value('detection_select_threshold'))

        # GPU list
        self._gpu_list = parse_gpu_list(self.config_value('gpu_list'))

        # Augmentation variables
        self._rotational_shifts = int(self.config_value('rotational_shifts'))
        self._resize_factor = float(self.config_value('resize_factor'))
        self._int_shift_factor = float(self.config_value('int_shift_factor'))

        # Read in database pickle
        self._use_hist = ( self.config_value('use_historical_database') == 'true' )
        self._negative_sample_count = int( self.config_value('negative_sample_count') )
        if self._use_hist:
            tree_fn = self.config_value('historical_tree')
            database_fn = self.config_value('historical_database')
            with open(tree_fn, "rb") as input_file:
                self._hist_tree = pickle.load(input_file)
            with open(database_fn, "rb") as input_file:
                self._hist_data = pickle.load(input_file)

        # Resnet model config
        resnet_img_size = int(self.config_value('resnet_model_input_size'))
        resnet_batch_size = int(self.config_value('resnet_batch_size'))
        resnet_model_path = self.config_value('resnet_model_path')

        self._app_feature_extractor = AugmentedResnetFeatureExtractor(resnet_model_path,
          resnet_img_size, resnet_batch_size, self._gpu_list, self._rotational_shifts,
          self._resize_factor,self._int_shift_factor)

        # Init variables
        self._grid = Grid()
        self._desc_counter = 0;

        # Finalize function
        self._base_configure()

    # ------------------------------------------------------------------------------------
    def _get_uid(self, aug_type):
        self._desc_counter = self._desc_counter + 1
        return "augmentation_" + aug_type + "_" + str( self._desc_counter )

    # ------------------------------------------------------------------------------------
    def _step(self):
        try:
            # Grab image container from port using traits
            in_img_c = self.grab_input_using_trait('image')
            dos_ptr = self.grab_input_using_trait('detected_object_set')

            # Declare outputs
            new_positive_descriptors = []
            new_positive_ids = []
            new_negative_descriptors = []
            new_negative_ids = []

            # Get detection bbox
            dos = dos_ptr.select(self._select_threshold)
            bbox_num = dos.size()

            # Make sure we have at least some detections
            app_f_begin = timer()
            if bbox_num != 0:
                im = get_pil_image(in_img_c.image())
                self._app_feature_extractor.frame = im
                pt_app_features = self._app_feature_extractor(dos) 
                for item in pt_app_features:
                    new_positive_descriptors.append(item.numpy())
                    new_positive_ids.append(self._get_uid("scale_and_shift"))

            app_f_end = timer()
            print('%%%aug app feature eclapsed time: {}'.format(app_f_end - app_f_begin))

            # Get negative descriptors via distance function
            neg_per_pos = int( self._negative_sample_count / len( new_positive_descriptors ) )

            if neg_per_pos <= 0:
                neg_per_pos = 1

            if self._use_hist:
                for nd in new_positive_descriptors:
                    # Do distance func in database
                    output, indxs = self._hist_tree.query( np.reshape( nd, np.size( nd ) ),
                                                            k=neg_per_pos )
                    # Make new negative entries
                    for ind in indxs:
                        new_negative_descriptors.append( self._hist_data[ ind ] )
                        new_negative_ids.append( self._get_uid("db_neg") )

            # push outputs
            vital_pos_descriptors = to_vital( new_positive_descriptors )
            vital_neg_descriptors = to_vital( new_negative_descriptors )

            self.push_to_port_using_trait('new_positive_descriptors', vital_pos_descriptors)
            self.push_to_port_using_trait('new_positive_ids', datum.VectorString(new_positive_ids))
            self.push_to_port_using_trait('new_negative_descriptors', vital_neg_descriptors)
            self.push_to_port_using_trait('new_negative_ids', datum.VectorString(new_negative_ids))
            self._base_step()

        except BaseException as e:
            print( repr( e ) )
            import traceback
            print( traceback.format_exc() )
            sys.stdout.flush()
            raise

    def __del__(self):
        print('!!!!Deleting augmentation python process!!!!')

# ==================================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch.resnet_augmentation'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('resnet_augmentation',
                                'Pytorch-Based Augmentation',
                                DataAugmentation)

    process_factory.mark_process_module_as_loaded(module_name)

