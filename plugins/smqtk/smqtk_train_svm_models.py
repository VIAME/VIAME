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

import smqtk.algorithms
import smqtk.iqr
import smqtk.representation
import smqtk.representation.descriptor_element.local_elements
import smqtk.utils.plugin

import json
import svmutil
import svm
import os
import ctypes
import filecmp
import sys
import numpy

def generate_svm_model( input_folder,
                        smqtk_params,
                        target_id = "positive",
                        id_extension = "lbl" ):

  ## Member variables to be configured in ``_configure``.
  di_json_config_path = None
  di_json_config = None

  # Path to the json config file for the NearestNeighborsIndex
  nn_json_config_path = None
  nn_json_config = None

  # Number of top, refined descriptor UUIDs to return per step.
  query_return_n = None

  # Set of descriptors to pull positive/negative querys from.
  descriptor_set = None

  # Nearest Neighbors index to use for IQR working index population.
  neighbor_index = None

  # IQR session state object
  iqr_session = None

  # Factory for converting vital descriptors IDs to SMQTK
  smqtk_descriptor_element_factory = smqtk.representation.DescriptorElementFactory(
      smqtk.representation.descriptor_element.local_elements.DescriptorMemoryElement,
      {}
  )

  di_json_config_path = smqtk_params('descriptor_index_config_file')
  nn_json_config_path = smqtk_params('neighbor_index_config_file')
  pos_seed_neighbors = int(smqtk_params('pos_seed_neighbors'))
  query_return_n = int(smqtk_params('query_return_size'))

  # parse json files
  with open(di_json_config_path) as f:
      di_json_config = json.load(f)
  with open(nn_json_config_path) as f:
      nn_json_config = json.load(f)

  descriptor_set = smqtk.utils.plugin.from_plugin_config(
      di_json_config,
      smqtk.representation.get_descriptor_index_impls()
  )
  neighbor_index = smqtk.utils.plugin.from_plugin_config(
      nn_json_config,
      smqtk.algorithms.get_nn_index_impls()
  )

  # Using default relevancy index configuration, which as of 2017/08/24
  # is the only one: libSVM-based relevancy ranking.
  iqr_session = smqtk.iqr.IqrSession( pos_seed_neighbors )

  print( "Stepping SMQTK Query Process" )
  #
  # Grab input values from ports using traits.
  #
  # Set/vector of descriptors to perform query off of
  #
  #: :type: vital.types.DescriptorSet
  vital_positive_descriptor_set = grab_input_using_trait('positive_descriptor_set')
  vital_positive_descriptor_uids = grab_input_using_trait('positive_exemplar_uids')
  #
  # Set/vector of descriptors to use as negative examples
  #
  #: :type: vital.types.DescriptorSet
  vital_negative_descriptor_set = grab_input_using_trait('negative_descriptor_set')
  vital_negative_descriptor_uids = grab_input_using_trait('negative_exemplar_uids')
  #
  # Vector of UIDs for vector of descriptors in descriptor_set.
  #
  #: :type: list[str]
  iqr_positive_tuple = grab_input_using_trait('iqr_positive_uids')
  iqr_negative_tuple = grab_input_using_trait('iqr_negative_uids')
  #
  # Optional input SVM model
  #
  #: :type: vital.types.UCharVector
  iqr_query_model = grab_input_using_trait('iqr_query_model')

  # Reset index on new query, a new query is one without IQR feedback
  if len( iqr_positive_tuple ) == 0 and len( iqr_negative_tuple ) == 0:
    iqr_session = smqtk.iqr.IqrSession(pos_seed_neighbors)

  # Convert descriptors to SMQTK elements.
  #: :type: list[DescriptorElement]
  user_pos_elements = []
  z = zip(vital_positive_descriptor_set.descriptors(), vital_positive_descriptor_uids)
  for vital_descr, uid_str in z:
      smqtk_descr = smqtk_descriptor_element_factory.new_descriptor(
          'from_sprokit', uid_str
      )
      # A descriptor may already exist in the backend (if its persistant)
      # for the given UID. We currently always overwrite.
      smqtk_descr.set_vector(vital_descr.todoublearray())
      # Queue up element for adding to set.
      user_pos_elements.append(smqtk_descr)

  user_neg_elements = []
  z = zip(vital_negative_descriptor_set.descriptors(), vital_negative_descriptor_uids)
  for vital_descr, uid_str in z:
      smqtk_descr = smqtk_descriptor_element_factory.new_descriptor(
          'from_sprokit', uid_str
      )
      # A descriptor may already exist in the backend (if its persistant)
      # for the given UID. We currently always overwrite.
      smqtk_descr.set_vector(vital_descr.todoublearray())
      # Queue up element for adding to set.
      user_neg_elements.append(smqtk_descr)

  # Get SMQTK descriptor elements from index for given pos/neg UUID-
  # values.
  #: :type: collections.Iterator[DescriptorElement]
  pos_descrs = descriptor_set.get_many_descriptors(iqr_positive_tuple)
  #: :type: collections.Iterator[DescriptorElement]
  neg_descrs = descriptor_set.get_many_descriptors(iqr_negative_tuple)

  iqr_session.adjudicate(user_pos_elements, user_neg_elements)
  iqr_session.adjudicate(set(pos_descrs), set(neg_descrs))

  # Update iqr working index for any new positives
  iqr_session.update_working_index(neighbor_index)

  iqr_session.refine()

  ordered_results = iqr_session.ordered_results()
  ordered_feedback = iqr_session.ordered_feedback()
  if query_return_n > 0:
      if ordered_feedback is not None:
          ordered_feedback_results = ordered_feedback[:query_return_n]
      else:
          ordered_feedback_results = []
      ordered_results = ordered_results[:query_return_n]

  return_elems, return_dists = zip(*ordered_results)
  return_uuids = [e.uuid() for e in return_elems]
  ordered_feedback_uuids = [e[0].uuid() for e in ordered_feedback_results]
  ordered_feedback_distances = [e[1] for e in ordered_feedback_results]
  # Just creating the scores to preserve the order in case of equal floats
  # because we're passing the distances explicity anyway
  ordered_feedback_scores = numpy.linspace(1, 0,
      len(ordered_feedback_distances))

  # Retrive IQR model from class
  try:
    return_model = get_svm_bytes()
  except:
    return_model = []


    def get_svm_bytes(self):
        svm_model = iqr_session.rel_index.get_model()
        tmp_file_name = "tmp_svm.model"

        svmutil.svm_save_model(tmp_file_name.encode(), svm_model)
        with open(tmp_file_name, "rb") as f:
            model_file = f.read()
            b = bytearray(model_file)
        os.remove(tmp_file_name)
        return b

    def get_model_from_bytes(self, bytes):
        c_bytes = (ctypes.c_ubyte * len(bytes))(*bytes)
        model = svmutil.svm_load_model_from_bytes(c_bytes)
        return model

    # (TODO (Mmanu)) Remove, intended for testing the code!
    def test_model_from_byte(self):
        # The original model
        svm_model_1 = iqr_session.rel_index.get_model()
        model_1_file, model_2_file = "tmp_svm_1.model", "tmp_svm_2.model"
        svmutil.svm_save_model(model_1_file.encode(), svm_model_1)

        # Get the bytes for the model first.
        bytes = get_svm_bytes()
        # Use the bytes to created a model
        svm_model_2 = get_model_from_bytes(bytes)
        # Save the model created using the bytes
        svmutil.svm_save_model(model_2_file.encode(), svm_model_2)

        # Check that the model created using the bytes is the same as the
        # original model.
        assert(filecmp.cmp(model_1_file, model_2_file) is True)
        os.remove(model_1_file)
        os.remove(model_2_file)

def generate_svm_models( input_folder,
                         smqtk_params = dict(),
                         id_extension = "lbl",
                         background_id = "background",
                         output_folder = "category_models" ):

  # Find all label files in input folder except background
  []

  # Error checking
  []

  # Generate output folder
  []

  # Generate SVM model for each category
  []
