# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from PIL import Image as pil_image
from random import randint

from viame.processes.base import ViameProcess
from viame.pipeline import process

from viame.types import Image
from viame.types import ImageContainer
from viame.types import DetectedObject, DetectedObjectSet
from viame.types import ObjectTrackState, Track, ObjectTrackSet

from viame.util.pil import get_pil_image, from_pil

import numpy as np

class blank_out_frames( ViameProcess ):
    """
    This process blanks out images which don't have detections on them.
    """
    # -------------------------------------------------------------------------
    def __init__( self, conf ):
        ViameProcess.__init__( self, conf )

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        #  declare our ports (port-name, flags)
        self.declare_input_port_using_trait( 'image', required )
        self.declare_input_port_using_trait( 'object_track_set', required )

        self.declare_output_port_using_trait( 'image', optional )

    # -------------------------------------------------------------------------
    def _configure( self ):
        self._base_configure()

    # -------------------------------------------------------------------------
    def _step( self ):
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait( 'image' )
        tracks = self.grab_input_using_trait( 'object_track_set' )

        # Get python image from conatiner (just for show)
        in_img = get_pil_image( in_img_c.image() ).convert( 'RGB' )

        if len( tracks.tracks() ) == 0:
          # Fill image
          in_img = pil_image.new( mode='RGB', size=in_img.size,
            color = ( randint( 0, 255 ), randint( 0, 255 ), randint( 0, 255 ) ) )

        # push dummy image object (same as input) to output port
        self.push_to_port_using_trait( 'image', ImageContainer( from_pil( in_img ) ) )

        self._base_step()

class percentile_norm_npy_16_to_8bit( ViameProcess ):
    """
    Percentile normalization on 16-bit input image, output to 8-bit numpy edition
    """
    # -------------------------------------------------------------------------
    def __init__( self, conf ):
        ViameProcess.__init__( self, conf )

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        #  declare our ports (port-name, flags)
        self.declare_input_port_using_trait( 'image', required )
        self.declare_output_port_using_trait( 'image', optional )

    # -------------------------------------------------------------------------
    def _configure( self ):
        self._base_configure()

    # -------------------------------------------------------------------------
    def _step( self ):
        # grab image container from port using traits
        img_c = self.grab_input_using_trait( 'image' )

        img = img_c.image().asarray().astype( "uint16" )

        mi = np.percentile( img, 1 )
        ma = np.percentile( img, 100 )

        normalized = ( img - mi ) / ( ma - mi )

        normalized = normalized * 255
        normalized[ normalized < 0 ] = 0

        output = ImageContainer( Image( normalized.astype( "uint8" ) ) )

         # push dummy image object (same as input) to output port
        self.push_to_port_using_trait( 'image', output )
        self._base_step()
