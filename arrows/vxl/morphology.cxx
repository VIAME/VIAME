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

// ----------------------------------------------------------------------------
inline bool
union_functor( bool x1, bool x2 )
{
  return x1 || x2;
}

// ----------------------------------------------------------------------------
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
  using morphology_func_t = void ( * )( vil_image_view< bool > const&,
                                        vil_image_view< bool >&,
                                        vil_structuring_element const& );

  // Set up structuring elements
  void setup_internals();

  // Compute the morphological operation on an image
  void apply_morphology( vil_image_view< bool > const& input,
                         vil_image_view< bool >& output );
  void apply_morphology( vil_image_view< bool > const& input,
                         vil_image_view< bool >& output,
                         morphology_func_t func );

  // Perform a morphological operation and optionally combine across channels
  vil_image_view< bool >
  perform_morphological_operations( vil_image_view< bool > const& input );

  bool configured{ false };
  vil_structuring_element morphological_element;

  morphology_mode morphology_type{ MORPHOLOGY_dilate };
  element_mode element_type{ ELEMENT_disk };
  combine_mode combine_type{ COMBINE_none };
  double kernel_radius{ 1.5 };
};

// ----------------------------------------------------------------------------
void
morphology::priv
::setup_internals()
{
  if( !configured )
  {
    switch( element_type )
    {
      case ELEMENT_disk:
      {
        morphological_element.set_to_disk( kernel_radius );
        break;
      }
      case ELEMENT_iline:
      {
        morphological_element.set_to_line_i(
            -static_cast< int >( kernel_radius ),
            static_cast< int >( kernel_radius ) );
        break;
      }
      case ELEMENT_jline:
      {
        morphological_element.set_to_line_j(
            -static_cast< int >( kernel_radius ),
            static_cast< int >( kernel_radius ) );
        break;
      }
    }

    configured = true;
  }
}

// ----------------------------------------------------------------------------
void
morphology::priv
::apply_morphology( vil_image_view< bool > const& input,
                    vil_image_view< bool >& output,
                    morphology_func_t func )
{
  for( auto plane_index : vital::range::iota( input.nplanes() ) )
  {
    auto input_plane = vil_plane( input, plane_index );
    auto output_plane = vil_plane( output, plane_index );
    func( input_plane, output_plane, morphological_element );
  }
}

// ----------------------------------------------------------------------------
void
morphology::priv
::apply_morphology( vil_image_view< bool > const& input,
                    vil_image_view< bool >& output )
{
  switch( morphology_type )
  {
    case MORPHOLOGY_erode:
    {
      this->apply_morphology( input, output, vil_binary_erode );
      break;
    }
    case MORPHOLOGY_dilate:
    {
      this->apply_morphology( input, output, vil_binary_dilate );
      break;
    }
    case MORPHOLOGY_close:
    {
      this->apply_morphology( input, output, vil_binary_closing );
      break;
    }
    case MORPHOLOGY_open:
    {
      this->apply_morphology( input, output, vil_binary_opening );
      break;
    }
    case MORPHOLOGY_none:
    {
      output.deep_copy( input );
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

  apply_morphology( input, output );

  if( combine_type == COMBINE_none )
  {
    // Don't combine across channels
    return output;
  }

  // Select whether to do pixel-wise union or intersection
  auto functor =
    ( combine_type == COMBINE_union
      ? union_functor : intersection_functor );

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
  config->set_value( "kernel_radius", d->kernel_radius,
                     "Radius of morphological kernel." );
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

  d->morphology_type =
    config->get_enum_value< morphology_converter >( "morphology" );
  d->element_type =
    config->get_enum_value< element_converter >( "element_shape" );
  d->kernel_radius = config->get_value< double >( "kernel_radius" );
  d->combine_type =
    config->get_enum_value< combine_converter >( "channel_combination" );

  // Note that some internal elements must be reset
  d->configured = false;
}

// ----------------------------------------------------------------------------
bool
morphology
::check_configuration( vital::config_block_sptr config ) const
{
  auto const kernel_radius = config->get_value< double >( "kernel_radius" );
  if( kernel_radius < 0 )
  {
    LOG_ERROR(
      logger(),
      "Config item kernel_radius should have been non-negative but was" <<
        kernel_radius );
  }
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
morphology
::filter( kwiver::vital::image_container_sptr image_data )
{
  // Perform basic validation
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

} // namespace vxl

} // namespace arrows

} // namespace kwiver
