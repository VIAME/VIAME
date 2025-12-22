/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Warp Detections Process
 */

#include "WarpDetectionsProcess.h"
#include "RegisterOpticalAndThermal.h"

#include <vital/vital_types.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/types/detected_object_set.h>

#include <itkTransformFileReader.h>

#include <exception>


namespace viame
{

namespace itk
{

create_config_trait( transformation_file, kwiver::vital::path_t, "",
  "Filename for the file containing an ITK composite transformation" );

//------------------------------------------------------------------------------
// Private implementation class
class itk_warp_detections_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  kwiver::vital::path_t m_transformation_file;
  NetTransformType::Pointer m_transformation;
};

// =============================================================================

itk_warp_detections_process
::itk_warp_detections_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new itk_warp_detections_process::priv() )
{
  make_ports();
  make_config();
}


itk_warp_detections_process
::~itk_warp_detections_process()
{
}


// -----------------------------------------------------------------------------
void
itk_warp_detections_process
::_configure()
{
  d->m_transformation_file = config_value_using_trait( transformation_file );

  ::itk::TransformFileReaderTemplate< TransformFloatType >::Pointer reader =
    ::itk::TransformFileReaderTemplate< TransformFloatType >::New();

  reader->SetFileName( d->m_transformation_file );
  reader->Update();

  if( reader->GetTransformList()->size() != 1 )
  {
    throw std::runtime_error( "Unable to load: " + d->m_transformation_file );
  }

  d->m_transformation = static_cast< NetTransformType* >(
    reader->GetTransformList()->begin()->GetPointer() );
}


// -----------------------------------------------------------------------------
void
itk_warp_detections_process
::_step()
{
  kwiver::vital::detected_object_set_sptr input;
  kwiver::vital::detected_object_set_sptr output;

  input = grab_from_port_using_trait( detected_object_set );

  try
  {
    if( input )
    {
      output = input->clone();

      for( auto detection : *output )
      {
        // Adjust detection box
        TransformFloatType lower_left[2] = {
          detection->bounding_box().min_x(), detection->bounding_box().min_y() };
        TransformFloatType lower_right[2] = {
          detection->bounding_box().max_x(), detection->bounding_box().min_y() };
        TransformFloatType upper_right[2] = {
          detection->bounding_box().max_x(), detection->bounding_box().max_y() };
        TransformFloatType upper_left[2] = {
          detection->bounding_box().min_x(), detection->bounding_box().max_y() };

        NetTransformType::OutputPointType pt1 =
          d->m_transformation->TransformPoint(
            NetTransformType::InputPointType( lower_left ) );
        NetTransformType::OutputPointType pt2 =
          d->m_transformation->TransformPoint(
            NetTransformType::InputPointType( lower_right ) );
        NetTransformType::OutputPointType pt3 =
          d->m_transformation->TransformPoint(
            NetTransformType::InputPointType( upper_right ) );
        NetTransformType::OutputPointType pt4 =
          d->m_transformation->TransformPoint(
            NetTransformType::InputPointType( upper_left ) );

        lower_left[0] = std::min( { pt1[0], pt2[0], pt3[0], pt4[0] } );
        lower_left[1] = std::min( { pt1[1], pt2[1], pt3[1], pt4[1] } );
        upper_right[0] = std::max( { pt1[0], pt2[0], pt3[0], pt4[0] } );
        upper_right[1] = std::max( { pt1[1], pt2[1], pt3[1], pt4[1] } );

        detection->set_bounding_box(
          kwiver::vital::bounding_box_d(
            lower_left[0], lower_left[1], upper_right[0], upper_right[1] ) );
      }
    }
  }
  catch( ... )
  {
    push_to_port_using_trait( detected_object_set,
      kwiver::vital::detected_object_set_sptr() );
    return;
  }

  push_to_port_using_trait( detected_object_set, output );
}


// -----------------------------------------------------------------------------
void
itk_warp_detections_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, required );
}


// -----------------------------------------------------------------------------
void
itk_warp_detections_process
::make_config()
{
  declare_config_using_trait( transformation_file );
}


// =============================================================================
itk_warp_detections_process::priv
::priv()
  : m_transformation_file( "" )
{
}


itk_warp_detections_process::priv
::~priv()
{
}

} // end namespace itk

} // end namespace viame
