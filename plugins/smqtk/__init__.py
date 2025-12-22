# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from viame.processes.smqtk import smqtk_ingest_descriptors
from viame.processes.smqtk import smqtk_process_query
from viame.processes.smqtk import smqtk_object_track_descriptors

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:smqtk.smqtk_processes'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process(
        'smqtk_ingest_descriptors',
        'Add descriptors and parallel UUIDs to a SMQTK descriptor index',
        smqtk_ingest_descriptors.SmqtkIngestDescriptors
    )

    process_factory.add_process(
        'smqtk_process_query',
        'Perform queries against some arbitrary descriptor index',
        smqtk_process_query.SmqtkProcessQuery
    )

    process_factory.add_process(
        'smqtk_object_track_descriptors',
        'Add descriptors to object tracks',
        smqtk_object_track_descriptors.SmqtkObjectTrackDescriptors
    )

    process_factory.mark_process_module_as_loaded(module_name)
