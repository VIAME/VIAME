# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

import logging

from vital.algo import ImageObjectDetector

logger = logging.getLogger(__name__)

from vital.types import Image
from vital.types import ImageContainer
from vital.types import DetectedObject
from vital.types import DetectedObjectSet
from vital.types import BoundingBox

class @template@Detector( ImageObjectDetector ):
    """
    Implementation of ImageObjectDetector class
    """
    def __init__( self ):
        ImageObjectDetector.__init__( self )

        # TODO: Keep these config variables or make new ones
        self._net_config = ""
        self._weight_file = ""
        self._class_names = ""

    def get_configuration(self):
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

        # TODO: do something with numpy image producing detections
        bboxes = []
        labels = []

        # Convert detections to kwiver format
        output = DetectedObjectSet()

        for bbox, label in zip( bboxes, labels ):

            bbox_int = bbox.astype( np.int32 )

            bounding_box = BoundingBox( bbox_int[0], bbox_int[1],
                                        bbox_int[2], bbox_int[3] )

            detected_object_type = DetectedObjectType( label, 1.0 )

            detected_object = DetectedObject( bounding_box,
                                              np.max( class_confidence ),
                                              detected_object_type )

            output.add( detected_object )

        return output

def __vital_algorithm_register__():
    from vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "@template@"

    if algorithm_factory.has_algorithm_impl_name(
      @template@Detector.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name,
      "@template@ dection inference routine", @template@Detector )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
