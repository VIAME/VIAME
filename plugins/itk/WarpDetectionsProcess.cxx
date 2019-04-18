/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
