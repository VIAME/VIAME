# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import MergeDetections

from kwiver.vital.types import DetectedObjectSet

##############################################################################
# Concatenation, the behavior merge_detection_sets had before it could run a
# merger algorithm. Every detection from every input is emitted unchanged, so
# overlaps are left for a downstream refiner to resolve.
##############################################################################

class MergeDetectionsSimple( MergeDetections ):
    """
    Implementation of MergeDetections class
    """
    def __init__( self ):
        MergeDetections.__init__( self )

    def get_configuration( self ):
        return super( MergeDetections, self ).get_configuration()

    def set_configuration( self, cfg_in ):
        pass

    def check_configuration( self, cfg ):
        return True

    def merge( self, det_sets ):
        output = DetectedObjectSet()

        for det_set in det_sets:
            if det_set is None:
                continue
            for det in det_set:
                output.add( det )

        return output

# Legacy name for the same behavior, used by the habcam pipelines.
class MergeDetectionsMerge( MergeDetectionsSimple ):
    """
    Implementation of MergeDetections class
    """
    pass

def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        MergeDetectionsSimple,
        "simple",
        "Concatenate all input detection sets without resolving overlaps",
    )

    register_vital_algorithm(
        MergeDetectionsMerge,
        "merge",
        "Concatenate all input detection sets without resolving overlaps",
    )
