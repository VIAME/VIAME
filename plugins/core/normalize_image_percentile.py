# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageFilter

from kwiver.vital.types import Image
from kwiver.vital.types import ImageContainer

import numpy as np

class PercentileNormalization( ImageFilter ):
  """
  Percentile-based image normalization filter.

  This filter normalizes image intensity values using percentile-based
  min/max calculation. It supports arbitrary input types and can output
  either 8-bit or native format (same as input with full range stretch).

  Configuration:
    - lower_percentile: Lower percentile for min value (default: 1.0)
    - upper_percentile: Upper percentile for max value (default: 100.0)
    - output_format: "8-bit" for 8-bit output, "native" for same as input (default: "8-bit")
  """
  def __init__( self ):
    ImageFilter.__init__( self )
    self.lower_percentile = 1.0
    self.upper_percentile = 100.0
    self.output_format = "8-bit"

  def get_configuration( self ):
    cfg = super( ImageFilter, self ).get_configuration()
    cfg.set_value( "lower_percentile", str( self.lower_percentile ) )
    cfg.set_value( "upper_percentile", str( self.upper_percentile ) )
    cfg.set_value( "output_format", self.output_format )
    return cfg

  def set_configuration( self, cfg_in ):
    self.lower_percentile = float( cfg_in.get_value( "lower_percentile" ) )
    self.upper_percentile = float( cfg_in.get_value( "upper_percentile" ) )
    self.output_format = cfg_in.get_value( "output_format" )

  def check_configuration( self, cfg ):
    lower = float( cfg.get_value( "lower_percentile" ) )
    upper = float( cfg.get_value( "upper_percentile" ) )
    output_fmt = cfg.get_value( "output_format" )

    if lower < 0.0 or lower > 100.0:
      print( "Error: lower_percentile must be between 0.0 and 100.0" )
      return False

    if upper < 0.0 or upper > 100.0:
      print( "Error: upper_percentile must be between 0.0 and 100.0" )
      return False

    if lower >= upper:
      print( "Error: lower_percentile must be less than upper_percentile" )
      return False

    if output_fmt not in [ "8-bit", "native" ]:
      print( "Error: output_format must be '8-bit' or 'native'" )
      return False

    return True

  def filter( self, in_img ):
    img = in_img.image().asarray()
    input_dtype = img.dtype

    # Convert to float64 for calculations
    img_float = img.astype( np.float64 )

    # Calculate percentiles
    p_low = np.percentile( img_float, self.lower_percentile )
    p_high = np.percentile( img_float, self.upper_percentile )

    # Avoid division by zero
    range_val = p_high - p_low
    if range_val <= 0:
      range_val = 1.0

    # Normalize to [0, 1]
    normalized = ( img_float - p_low ) / range_val
    normalized = np.clip( normalized, 0.0, 1.0 )

    # Determine output format and scale accordingly
    if self.output_format == "8-bit":
      output = ( normalized * 255.0 ).astype( np.uint8 )
    else:
      # Native format - stretch to full range of input type
      if input_dtype == np.uint8:
        output = ( normalized * 255.0 ).astype( np.uint8 )
      elif input_dtype == np.uint16:
        output = ( normalized * 65535.0 ).astype( np.uint16 )
      elif input_dtype == np.int16:
        output = ( normalized * 65535.0 - 32768.0 ).astype( np.int16 )
      elif input_dtype == np.float32:
        output = normalized.astype( np.float32 )
      elif input_dtype == np.float64:
        output = normalized
      else:
        # Fallback: try to preserve dtype, scale to 8-bit range
        output = ( normalized * 255.0 ).astype( input_dtype )

    return ImageContainer( Image( output ) )

def __vital_algorithm_register__():
  from kwiver.vital.algo import algorithm_factory

  # Register Algorithm
  implementation_name  = "percentile_norm_npy"
  if algorithm_factory.has_algorithm_impl_name(
      PercentileNormalization.static_type_name(), implementation_name ):
    return

  algorithm_factory.add_algorithm( implementation_name,
    "Numpy percentile normalization with configurable output format", PercentileNormalization )

  algorithm_factory.mark_algorithm_as_loaded( implementation_name )
