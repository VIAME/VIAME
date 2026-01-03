# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import json

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process

from kwiver.vital.types import new_descriptor, DescriptorSet

# Use local smqtk package for VIAME
from .smqtk import representation as smqtk_representation
from .smqtk.utils import plugin as smqtk_plugin

import psycopg2


class SmqtkObjectTrackDescriptors (KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # register python config file
        self.add_config_trait(
            'config_file', 'config_file', '',
            'Path to the json configuration file for the descriptor index to fetch from.'
        )
        self.declare_config_using_trait('config_file')

        self.add_config_trait(
            'video_name', 'video_name', '',
            'Video name'
        )
        self.declare_config_using_trait('video_name')

        self.add_config_trait(
            'conn_str', 'conn_str', '',
            'Connection string'
        )
        self.declare_config_using_trait('conn_str')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('object_track_set', required)
        self.declare_output_port_using_trait('object_track_set', optional)

    def _configure(self):
        self.config_file = self.config_value('config_file')
        self.video_name = self.config_value('video_name')
        self.conn_str = self.config_value('conn_str')

        # parse json file
        with open(self.config_file) as data_file:
            self.json_config = json.load(data_file)

        #: :type: smqtk_representation.DescriptorIndex
        self.smqtk_descriptor_index = smqtk_plugin.from_plugin_config(
            self.json_config,
            smqtk_representation.get_descriptor_index_impls()
        )

        self.conn = psycopg2.connect(self.conn_str)
        self.current_idx = 0

        self._base_configure()

    def _step(self):
        object_tracks = self.grab_input_using_trait('object_track_set')

        for object_track in object_tracks.tracks():
            for track_state in object_track:
                if track_state.frame_id == self.current_idx:
                    cur = self.conn.cursor()
                    cur.execute("SELECT track_descriptor.uid FROM track_descriptor "
                                "INNER JOIN track_descriptor_track ON track_descriptor.uid = track_descriptor_track.uid "
                                "INNER JOIN track_descriptor_history ON track_descriptor.uid = track_descriptor_history.uid "
                                "WHERE track_descriptor.video_name = %(video_name)s AND track_descriptor_history.frame_number = %(frame_number)s AND track_descriptor_track.track_id = %(track_id)s",
                                {
                                    "video_name": self.video_name,
                                    "frame_number": track_state.frame_id,
                                    "track_id": object_track.id,
                                })
                    rows = list(cur.fetchall())
                    if len(rows) != 1:
                        raise RuntimeError("Could not get track descriptor")
                    uid = rows[0][0]

                    smqtk_descriptor = self.smqtk_descriptor_index.get_descriptor(uid)

                    vital_descriptor = new_descriptor(len(smqtk_descriptor.vector()), "d")
                    vital_descriptor[:] = smqtk_descriptor.vector()

                    track_state.detection.set_descriptor(vital_descriptor)

                    print("Finished track state: %i %i" % (object_track.id, track_state.frame_id))

        self.push_to_port_using_trait('object_track_set', object_tracks)

        self.current_idx += 1

        self._base_step()
