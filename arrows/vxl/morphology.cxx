// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "morphology.h"

#include <arrows/vxl/image_container.h>

#include <vital/util/enum_converter.h>

#include <vital/range/iota.h>

#include <vil/algo/vil_binary_closing.h>
#include <vil/algo/vil_binary_dilate.h>
#include <vil/algo/vil_binary_erode.h>
#include <vil/algo/vil_binary_opening.h>
#include <vil/algo/vil_structuring_element.h>
#include <vil/vil_plane.h>
#include <vil/vil_transform.h>

namespace kwiver {

namespace arrows {

namespace vxl {

namespace {

enum morphology_mode
{
  MORPHOLOGY_erode,
  MORPHOLOGY_dilate,
  MORPHOLOGY_open,
  MORPHOLOGY_close,
  MORPHOLOGY_none,
};

ENUM_CONVERTER( morphology_converter, morphology_mode,
                { "erode", MORPHOLOGY_erode }, { "dilate", MORPHOLOGY_dilate },
                { "open", MORPHOLOGY_open }, { "close", MORPHOLOGY_close },
                { "none", MORPHOLOGY_none } );

enum element_mode
{
  ELEMENT_disk,
  ELEMENT_jline,
  ELEMENT_iline,
};

ENUM_CONVERTER( element_converter, element_mode, { "disk", ELEMENT_disk },
                { "iline", ELEMENT_iline }, { "jline", ELEMENT_jline } );

enum combine_mode
{
  COMBINE_none,
  COMBINE_union,
  COMBINE_intersection,
};

ENUM_CONVERTER( combine_converter, combine_mode, { "none", COMBINE_none },
                { "union", COMBINE_union },
                { "intersection", COMBINE_intersection } );

inline bool
union_functor( bool x1, bool x2 )
{
  return x1 || x2;
}

inline bool
intersection_functor( bool x1, bool x2 )
{
  return x1 && x2;
}

} // namespace

// ----------------------------------------------------------------------------
class morphology::priv
{
public:
  priv()
  {
  }

  ~priv();
  // Setup structuring elements
  void setup_internals();
  // Compute the morpholigical operation on a single plane of an input image
  void
  apply_morphology_to_plane( vil_image_view< bool > const& input,
                             vil_image_view< bool >& output,
                             unsigned plane_index );
  // Perform a morphological operation and optionally combine across channels
  vil_image_view< bool >
  perform_morphological_operations( vil_image_view< bool > const& input );

  bool setup{ false };
  vil_structuring_element morphological_element;

  morphology_mode morphology_type{ MORPHOLOGY_dilate };
  element_mode element_type{ ELEMENT_disk };
  combine_mode combine_type{ COMBINE_none };
  double element_size{ 1.0 };
};

// ----------------------------------------------------------------------------
morphology::priv
::~priv()
{
}

// ----------------------------------------------------------------------------
void
morphology::priv
::setup_internals()
{
  if( !setup )
  {
    switch( element_type )
    {
      case ELEMENT_disk:
      {
        morphological_element.set_to_disk( element_size );
      }
      case ELEMENT_iline:
      {
        morphological_element.set_to_line_i( 0, element_size );
      }
      case ELEMENT_jline:
      {
        morphological_element.set_to_line_j( 0, element_size );
      }
    }

    setup = true;
  }
}

// ----------------------------------------------------------------------------
void
morphology::priv
::apply_morphology_to_plane( vil_image_view< bool > const& input,
                             vil_image_view< bool >& output,
                             unsigned plane_index )
{
  auto input_plane = vil_plane( input, plane_index );
  auto output_plane = vil_plane( output, plane_index );
  switch( morphology_type )
  {
    case MORPHOLOGY_erode:
    {
      vil_binary_erode( input_plane, output_plane, morphological_element );
      break;
    }
    case MORPHOLOGY_dilate:
    {
      vil_binary_dilate( input_plane, output_plane, morphological_element );
      break;
    }
    case MORPHOLOGY_close:
    {
      vil_binary_closing( input_plane, output_plane, morphological_element );
      break;
    }
    case MORPHOLOGY_open:
    {
      vil_binary_opening( input_plane, output_plane, morphological_element );
      break;
    }
    case MORPHOLOGY_none:
    {
      output_plane.deep_copy( input_plane );
      break;
    }
  }
}

// ----------------------------------------------------------------------------
vil_image_view< bool >
morphology::priv
::perform_morphological_operations( vil_image_view< bool > const& input )
{
  setup_internals();

  vil_image_view< bool > output{ input.ni(), input.nj(), input.nplanes() };

  if( morphology_type != MORPHOLOGY_none )
  {
    for( auto i : vital::range::iota( input.nplanes() ) )
    {
      apply_morphology_to_plane( input, output, i );
    }
  }

  if( combine_type == COMBINE_none )
  {
    // Don't combine across channels
    return output;
  }

  // Select whether to do pixel-wise union or intersection
  auto functor =
    ( combine_type ==
      COMBINE_union ) ? union_functor : intersection_functor;

  auto accumulator = vil_plane( output, 0 );
  for( unsigned i = 1; i < output.nplanes(); ++i )
  {
    auto current_plane = vil_plane( output, i );
    // Union or intersect the current plane with the accumulator
    vil_transform( accumulator, current_plane, accumulator, functor );
  }
  return accumulator;
}

// ----------------------------------------------------------------------------
morphology
::morphology()
  : d{ new priv{} }
{
  attach_logger( "arrows.vxl.morphology" );
}

// ----------------------------------------------------------------------------
morphology::~morphology() = default;

// ----------------------------------------------------------------------------
vital::config_block_sptr
morphology
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value(
    "morphology", morphology_converter().to_string( d->morphology_type ),
    "Morphological operation to apply. Possible options are: " +
    morphology_converter().element_name_string() );
  config->set_value(
    "element_shape", element_converter().to_string( d->element_type ),
    "Shape of the structuring element. Possible options are: " +
    element_converter().element_name_string() );
  config->set_value( "element_size", d->element_size,
                     "Size of the structuring element." );
  config->set_value(
    "channel_combination", combine_converter().to_string( d->combine_type ),
    "Method for combining multiple binary channels. Possible options are: " +
    morphology_converter().element_name_string() );

  return config;
}

// ----------------------------------------------------------------------------
void
morphology
::set_configuration( vital::config_block_sptr in_config )
{
  // Start with our generated vital::config_block to ensure that assumed values
  // are present. An alternative would be to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d->morphology_type = config->get_enum_value< morphology_converter >(
    "morphology" );
  d->element_type = config->get_enum_value< element_converter >(
    "element_shape" );
  d->element_size = config->get_value< double >( "element_size" );
  d->combine_type = config->get_enum_value< combine_converter >(
    "channel_combination" );

  // Note that some internal elements must be reset
  d->setup = false;
}

// ----------------------------------------------------------------------------
bool
morphology
::check_configuration( vital::config_block_sptr config ) const
{
  auto const element_size = config->get_value< double >( "element_size" );
  if( element_size < 0 )
  {
    LOG_ERROR(
      logger(),
      "Config item element_size should have been non-negative but was" <<
        element_size );
  }
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
morphology
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform Basic Validation
  if( !image_data )
  {
    return nullptr;
  }

  // Get input image
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl( image_data->get_image() );

  if( view->pixel_format() != VIL_PIXEL_FORMAT_BOOL )
  {
    LOG_ERROR( logger(), "Input format must be a bool" );
    return nullptr;
  }

  auto filtered =
    d->perform_morphological_operations(
      static_cast< vil_image_view< bool > >( view ) );

  return std::make_shared< vxl::image_container >( filtered );
}

} // end namespace vxl

} // end namespace arrows

} // end namespace kwiver
