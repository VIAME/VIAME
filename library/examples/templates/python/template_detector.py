# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

import logging

import numpy as np

from viame.algo import ImageObjectDetector
from viame.types import (
    BoundingBoxD,
    DetectedObject,
    DetectedObjectSet,
    DetectedObjectType,
)

logger = logging.getLogger(__name__)


class @template@Detector( ImageObjectDetector ):
    """
    Implementation of ImageObjectDetector class

    Registered by the declaration in this package's `__init__.py`.
    """
    def __init__( self ):
        ImageObjectDetector.__init__( self )

        # TODO: Keep these config variables or make new ones
        self._net_config = ""
        self._weight_file = ""
        self._class_names = ""

    def get_configuration( self ):
        # Inherit from the base class
        cfg = super( ImageObjectDetector, self ).get_configuration()

        # TODO: Keep these config variables or make new ones
        cfg.set_value( "net_config", self._net_config )
        cfg.set_value( "weight_file", self._weight_file )
        cfg.set_value( "class_names", self._class_names )
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        # TODO: Keep these config variables or make new ones
        self._net_config = str( cfg.get_value( "net_config" ) )
        self._weight_file = str( cfg.get_value( "weight_file" ) )
        self._class_names = str( cfg.get_value( "class_names" ) )

    def check_configuration( self, cfg ):
        # TODO: Keep these config variables or make new ones
        if not cfg.has_value( "net_config" ):
            logger.error( "A network config file must be specified!" )
            return False
        if not cfg.has_value( "class_names" ):
            logger.error( "A class file must be specified!" )
            return False
        if not cfg.has_value( "weight_file" ):
            logger.error( "No weight file specified" )
            return False
        return True

    def detect( self, image_data ):
        # Convert image to 8-bit numpy
        input_image = image_data.asarray().astype( 'uint8' )

        # TODO: do something with numpy image producing detections, as
        # boxes (min_x, min_y, max_x, max_y), labels and confidences
        bboxes = []
        labels = []
        scores = []

        # Convert detections to kwiver format
        output = DetectedObjectSet()

        for bbox, label, score in zip( bboxes, labels, scores ):
            bounding_box = BoundingBoxD( float( bbox[0] ), float( bbox[1] ),
                                         float( bbox[2] ), float( bbox[3] ) )

            detected_object_type = DetectedObjectType( label, float( score ) )

            output.add( DetectedObject( bounding_box, float( score ),
                                        detected_object_type ) )

        return output
