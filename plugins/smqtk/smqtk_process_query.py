#ckwg +28
# Copyright 2017 by Kitware, Inc.
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

import itertools
import json

from sprokit.pipeline import process

from kwiver.kwiver_process import KwiverProcess

import smqtk.algorithms
import smqtk.iqr
import smqtk.representation
import smqtk.representation.descriptor_element.local_elements
import smqtk.utils.plugin


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
        self.declare_input_port_using_trait('descriptor_set', required)
        # UUIDs for the user provided positive exemplar descriptors
        self.declare_input_port_using_trait('exemplar_uids', required)
        # user adjudicated positive and negative descriptor UUIDs
        self.declare_input_port_using_trait('positive_uids', optional)
        self.declare_input_port_using_trait('negative_uids', optional)

        # Output, ranked descriptor UUIDs
        self.declare_output_port_using_trait('result_descriptor_uids', optional)
        # Output, ranked descriptor scores.
        self.declare_output_port_using_trait('result_descriptor_scores', optional)

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
        self.add_port_trait("exemplar_uids", "string_vector",
                            "Positive exemplar descriptor UUIDs")
        self.add_port_trait("positive_uids", "string_vector",
                            "Positive sample UIDs")
        self.add_port_trait("negative_uids", "string_vector",
                            "Negative sample UIDs")
        self.add_port_trait("result_descriptor_uids", "string_vector",
                            "Result ranked descriptor UUIDs in rank order.")
        self.add_port_trait("result_descriptor_scores", "double_vector",
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


    def _configure(self):
        self.di_json_config_path = self.config_value('descriptor_index_config_file')
        self.nn_json_config_path = self.config_value('neighbor_index_config_file')
        pos_seed_neighbors = int(self.config_value('pos_seed_neighbors'))
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
        self.iqr_session = smqtk.iqr.IqrSession(pos_seed_neighbors)

        self._base_configure()

    def _step(self):
        #
        # Grab input values from ports using traits.
        #
        # Set/vector of descriptors to perform query off of
        #
        #: :type: vital.types.DescriptorSet
        vital_descriptor_set = self.grab_input_using_trait('descriptor_set')
        vital_descriptor_uids = self.grab_input_using_trait('exemplar_uids')
        #
        # Vector of UIDs for vector of descriptors in descriptor_set.
        #
        #: :type: list[str]
        positive_tuple = self.grab_input_using_trait('positive_uids')
        negative_tuple = self.grab_input_using_trait('negative_uids')

        # Convert descriptors to SMQTK elements.
        #: :type: list[DescriptorElement]
        user_pos_elements = []
        z = itertools.izip(vital_descriptor_set.descriptors(), vital_descriptor_uids)
        for vital_descr, uid_str in z:
            smqtk_descr = self.smqtk_descriptor_element_factory.new_descriptor(
                'from_sprokit', uid_str
            )
            # A descriptor may already exist in the backend (if its persistant)
            # for the given UID. We currently always overwrite.
            smqtk_descr.set_vector(vital_descr.todoublearray())
            # Queue up element for adding to set.
            user_pos_elements.append(smqtk_descr)

        # Get SMQTK descriptor elements from index for given pos/neg UUID-
        # values.
        #: :type: collections.Iterator[DescriptorElement]
        pos_descrs = self.descriptor_set.get_many_descriptors(positive_tuple)
        #: :type: collections.Iterator[DescriptorElement]
        neg_descrs = self.descriptor_set.get_many_descriptors(negative_tuple)

        self.iqr_session.adjudicate(user_pos_elements)
        self.iqr_session.adjudicate(pos_descrs, neg_descrs)

        # Update iqr working index for any new positives
        self.iqr_session.update_working_index(self.neighbor_index)

        self.iqr_session.refine()

        ordered_results = self.iqr_session.ordered_results()
        if self.query_return_n > 0:
            ordered_results = ordered_results[:self.query_return_n]

        return_elems, return_dists = zip(*ordered_results)
        return_uuids = [e.uuid() for e in return_elems]

        # Pass on input descriptors and UIDs
        self.push_to_port_using_trait('result_descriptor_uids', return_uuids)
        self.push_to_port_using_trait('result_descriptor_scores', return_dists)

        self._base_step()
