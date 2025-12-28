# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
from __future__ import print_function

import logging
import json

from six.moves import zip

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.sprokit.pipeline import datum

import smqtk.algorithms
import smqtk.iqr
import smqtk.representation
import smqtk.representation.descriptor_element.local_elements
import smqtk.utils.plugin
import svmutil
import svm
import os
import ctypes
import filecmp

import sys
import threading
import traceback
import numpy

# Configure module logger with file handler for SMQTK queries
logger = logging.getLogger(__name__)

def _setup_file_logging():
    """Setup file-based logging for SMQTK query process."""
    import os
    log_dir = 'database/logs'
    if os.path.exists(log_dir):
        handler = logging.FileHandler(os.path.join(log_dir, 'SMQTK_Query_Log'))
        handler.setFormatter(logging.Formatter('[%(thread)d] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

_setup_file_logging()

class SmqtkProcessQuery (KwiverProcess):
    """
    Process for taking in a query descriptor set, alongside any known positive
    and negative UUIDs, converting them into SMQTK descriptor elements (variable
    backend), and performing a query off of them.

    """

    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # register port and config traits
        self.add_port_traits()
        self.add_config_traits()

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        ## declare our input port ( port-name,flags)
        # user-provided positive examplar descriptors.
        self.declare_input_port_using_trait('positive_descriptor_set', required)
        self.declare_input_port_using_trait('positive_exemplar_uids', required)
        # user-provided negative examplar descriptors.
        self.declare_input_port_using_trait('negative_descriptor_set', optional)
        self.declare_input_port_using_trait('negative_exemplar_uids', optional)
        # user adjudicated positive and negative descriptor UUIDs
        self.declare_input_port_using_trait('iqr_positive_uids', optional)
        self.declare_input_port_using_trait('iqr_negative_uids', optional)
        # input model if pre-generated
        self.declare_input_port_using_trait('iqr_query_model', optional)

        # Output, ranked descriptor UUIDs
        self.declare_output_port_using_trait('result_uids', optional)
        # Output, feedback descriptor UUIDs in decreasing order of 
        # importance
        self.declare_output_port_using_trait('feedback_uids', optional)
        # Output, ranked descriptor scores.
        self.declare_output_port_using_trait('result_scores', optional)
        # Output, distances of feedback descriptors from the separating plane
        # in increasing order of magnitude.  These distances correspond to
        # feedback_uids.
        self.declare_output_port_using_trait('feedback_distances', optional)
        # Output, feedback scores corresponding to the descriptors.
        self.declare_output_port_using_trait('feedback_scores', optional)
        # Output, trained IQR model.
        self.declare_output_port_using_trait('result_model', optional)

        ## Member variables to be configured in ``_configure``.
        # Path to the json config file for the DescriptorIndex
        self.di_json_config_path = None
        self.di_json_config = None
        # Path to the json config file for the NearestNeighborsIndex
        self.nn_json_config_path = None
        self.nn_json_config = None
        # Number of top, refined descriptor UUIDs to return per step.
        self.query_return_n = None
        # Set of descriptors to pull positive/negative querys from.
        self.descriptor_set = None
        # Nearest Neighbors index to use for IQR working index population.
        self.neighbor_index = None
        # IQR session state object
        self.iqr_session = None
        # Factory for converting vital descriptors to smqtk. Currently just
        # use in-memory elements for conversion.
        self.smqtk_descriptor_element_factory = smqtk.representation.DescriptorElementFactory(
            smqtk.representation.descriptor_element.local_elements.DescriptorMemoryElement,
            {}
        )

    def add_port_traits(self):
        self.add_port_trait("positive_descriptor_set", "descriptor_set",
                            "Positive exemplar descriptor set")
        self.add_port_trait("positive_exemplar_uids", "string_vector",
                            "Positive exemplar descriptor UUIDs")
        self.add_port_trait("negative_descriptor_set", "descriptor_set",
                            "Negative exemplar descriptor set")
        self.add_port_trait("negative_exemplar_uids", "string_vector",
                            "Negative exemplar descriptor UUIDs")
        self.add_port_trait("iqr_positive_uids", "string_vector",
                            "Positive sample UIDs")
        self.add_port_trait("iqr_negative_uids", "string_vector",
                            "Negative sample UIDs")
        self.add_port_trait("iqr_query_model", "uchar_vector",
                            "Input model for input queries.")
        self.add_port_trait("result_uids", "string_vector",
                            "Result ranked descriptor UUIDs in rank order.")
        self.add_port_trait("feedback_uids", "string_vector",
                            "Uuids of the descriptors in decreasing order of "
                            "importance for feedback")
        self.add_port_trait("result_scores", "double_vector",
                            "Result ranked descriptor distance score values "
                            "in rank order.")
        self.add_port_trait("feedback_distances", "double_vector",
                            "Distances from the separating plane in increasing"
                            "order of magnitude. The distances correspond to "
                            "feedback descriptors")
        self.add_port_trait("feedback_scores", "double_vector",
                            "Scores corresponding to the feedback descriptors")
        self.add_port_trait("result_model", "uchar_vector",
                            "Result ranked descriptor distance score values "
                            "in rank order.")

    def add_config_traits(self):
        # register python config file
        self.add_config_trait(
            'descriptor_index_config_file', 'descriptor_index_config_file', '',
            'Path to the json configuration file for the descriptor index.'
        )
        self.declare_config_using_trait('descriptor_index_config_file')

        self.add_config_trait(
            'neighbor_index_config_file', 'neighbor_index_config_file', '',
            'Path to the json configuration file for the nearest-neighbors '
            'algorithm configuration file.'
        )
        self.declare_config_using_trait('neighbor_index_config_file')

        self.add_config_trait(
            'pos_seed_neighbors', 'pos_seed_neighbors', '500',
            'Number of near neighbors to pull from the neighbor index for each'
            'positive example and adjudication when updating the working '
            'index.'
        )
        self.declare_config_using_trait('pos_seed_neighbors')

        self.add_config_trait(
            'query_return_size', 'query_return_size', '300',
            'The number of IQR ranked elements to return. If set to 0, we '
            'return the whole ranked set, which may become large over time.'
        )
        self.declare_config_using_trait('query_return_size')

    def get_svm_bytes(self):
        svm_model = self.iqr_session.rel_index.get_model()
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
        svm_model_1 = self.iqr_session.rel_index.get_model()
        model_1_file, model_2_file = "tmp_svm_1.model", "tmp_svm_2.model"
        svmutil.svm_save_model(model_1_file.encode(), svm_model_1)

        # Get the bytes for the model first.
        bytes = self.get_svm_bytes()
        # Use the bytes to created a model
        svm_model_2 = self.get_model_from_bytes(bytes)
        # Save the model created using the bytes
        svmutil.svm_save_model(model_2_file.encode(), svm_model_2)

        # Check that the model created using the bytes is the same as the
        # original model.
        assert(filecmp.cmp(model_1_file, model_2_file) is True)
        os.remove(model_1_file)
        os.remove(model_2_file)

    def _configure(self):
      try:
        logger.info( "Configuring SMQTK Query Process" )

        self.di_json_config_path = self.config_value('descriptor_index_config_file')
        self.nn_json_config_path = self.config_value('neighbor_index_config_file')
        self.pos_seed_neighbors = int(self.config_value('pos_seed_neighbors'))
        self.query_return_n = int(self.config_value('query_return_size'))

        # parse json files
        with open(self.di_json_config_path) as f:
            self.di_json_config = json.load(f)
        with open(self.nn_json_config_path) as f:
            self.nn_json_config = json.load(f)

        self.descriptor_set = smqtk.utils.plugin.from_plugin_config(
            self.di_json_config,
            smqtk.representation.get_descriptor_index_impls()
        )
        self.neighbor_index = smqtk.utils.plugin.from_plugin_config(
            self.nn_json_config,
            smqtk.algorithms.get_nn_index_impls()
        )

        # Using default relevancy index configuration, which as of 2017/08/24
        # is the only one: libSVM-based relevancy ranking.
        self.iqr_session = smqtk.iqr.IqrSession(self.pos_seed_neighbors)

        self._base_configure()

      except BaseException as e:
        logger.exception( "Error configuring SMQTK Query Process: %s", e )

    def _step(self):
      try:
        logger.debug( "Stepping SMQTK Query Process" )
        #
        # Grab input values from ports using traits.
        #
        # Set/vector of descriptors to perform query off of
        #
        #: :type: kwiver.vital.types.DescriptorSet
        vital_positive_descriptor_set = self.grab_input_using_trait('positive_descriptor_set')
        vital_positive_descriptor_uids = self.grab_input_using_trait('positive_exemplar_uids')
        #
        # Set/vector of descriptors to use as negative examples
        #
        #: :type: kwiver.vital.types.DescriptorSet
        vital_negative_descriptor_set = self.grab_input_using_trait('negative_descriptor_set')
        vital_negative_descriptor_uids = self.grab_input_using_trait('negative_exemplar_uids')
        #
        # Vector of UIDs for vector of descriptors in descriptor_set.
        #
        #: :type: list[str]
        iqr_positive_tuple = self.grab_input_using_trait('iqr_positive_uids')
        iqr_negative_tuple = self.grab_input_using_trait('iqr_negative_uids')
        #
        # Optional input SVM model
        #
        #: :type: kwiver.vital.types.UCharVector
        iqr_query_model = self.grab_input_using_trait('iqr_query_model')

        # Reset index on new query, a new query is one without IQR feedback
        if len( iqr_positive_tuple ) == 0 and len( iqr_negative_tuple ) == 0:
          self.iqr_session = smqtk.iqr.IqrSession(self.pos_seed_neighbors)

        # Convert descriptors to SMQTK elements.
        #: :type: list[DescriptorElement]
        user_pos_elements = []
        z = zip(vital_positive_descriptor_set.descriptors(), vital_positive_descriptor_uids)
        for vital_descr, uid_str in z:
            smqtk_descr = self.smqtk_descriptor_element_factory.new_descriptor(
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
            smqtk_descr = self.smqtk_descriptor_element_factory.new_descriptor(
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
        pos_descrs = self.descriptor_set.get_many_descriptors(iqr_positive_tuple)
        #: :type: collections.Iterator[DescriptorElement]
        neg_descrs = self.descriptor_set.get_many_descriptors(iqr_negative_tuple)

        self.iqr_session.adjudicate(user_pos_elements, user_neg_elements)
        self.iqr_session.adjudicate(set(pos_descrs), set(neg_descrs))

        # Update iqr working index for any new positives
        self.iqr_session.update_working_index(self.neighbor_index)

        self.iqr_session.refine()

        ordered_results = self.iqr_session.ordered_results()
        ordered_feedback = self.iqr_session.ordered_feedback()
        if self.query_return_n > 0:
            if ordered_feedback is not None:
                ordered_feedback_results = ordered_feedback[:self.query_return_n]
            else:
                ordered_feedback_results = []
            ordered_results = ordered_results[:self.query_return_n]

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
          return_model = self.get_svm_bytes()
        except Exception as e:
          logger.warning( "Failed to retrieve SVM model bytes: %s", e )
          return_model = []

        # Pass on input descriptors and UIDs
        self.push_to_port_using_trait('result_uids',
            datum.VectorString(return_uuids) )
        self.push_to_port_using_trait('feedback_uids',
            datum.VectorString(ordered_feedback_uuids))
        self.push_to_port_using_trait('result_scores',
            datum.VectorDouble(return_dists) )
        self.push_to_port_using_trait('feedback_distances',
            datum.VectorDouble(ordered_feedback_distances))
        self.push_to_port_using_trait('feedback_scores',
            datum.VectorDouble(ordered_feedback_scores))
        self.push_to_port_using_trait('result_model',
            datum.VectorUChar(return_model) )

        self._base_step()

      except BaseException as e:
        logger.exception( "Error in SMQTK Query Process step: %s", e )
