# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function

from kwiver.vital.algo import ImageFilter

from kwiver.vital.types import Image
from kwiver.vital.types import ImageContainer

import numpy as np

class PercentileNorm16BitTo8Bit( ImageFilter ):
  """
  Implementation of ImageFilter class
  """
  def __init__( self ):
    ImageFilter.__init__( self )

  def get_configuration( self ):
    cfg = super( ImageFilter, self ).get_configuration()
    return cfg

  def set_configuration( self, cfg_in ):
    return

  def check_configuration( self, cfg ):
    return True

  def filter( self, in_img ):
    img = in_img.image().asarray().astype( "uint16" )

    mi = np.percentile( img, 1 )
    ma = np.percentile( img, 100 )

    normalized = ( img - mi ) / ( ma - mi )
    normalized = normalized * 255
    normalized[ normalized < 0 ] = 0

    output = ImageContainer( Image( normalized.astype( "uint8" ) ) )
    return output

def __vital_algorithm_register__():
  from kwiver.vital.algo import algorithm_factory

  # Register Algorithm
  implementation_name  = "npy_percentile_norm"
  if algorithm_factory.has_algorithm_impl_name(
      PercentileNorm16BitTo8Bit.static_type_name(), implementation_name ):
    return

  algorithm_factory.add_algorithm( implementation_name,
    "Numpy percentile normalization", PercentileNorm16BitTo8Bit )

  algorithm_factory.mark_algorithm_as_loaded( implementation_name )
