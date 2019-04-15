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
 * \brief Warp Image Process
 */

#include "WarpImageProcess.h"
#include "RegisterOpticalAndThermal.h"

#include <vital/vital_types.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <arrows/ocv/image_container.h>

#include <vital/types/image_container.h>

#include <itkOpenCVImageBridge.h>
#include <itkTransformFileReader.h>

#include <exception>


namespace viame
{

namespace itk
{

create_config_trait( transformation_file, kwiver::vital::path_t, "",
  "Filename for the file containing an ITK composite transformation" );

create_port_trait( size_image, image, "Image to get output size from." );

//------------------------------------------------------------------------------
// Private implementation class
class itk_warp_image_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  kwiver::vital::path_t m_transformation_file;
  NetTransformType::Pointer m_transformation;
};

// =============================================================================

itk_warp_image_process
::itk_warp_image_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new itk_warp_image_process::priv() )
{
  make_ports();
  make_config();
}


itk_warp_image_process
::~itk_warp_image_process()
{
}


// -----------------------------------------------------------------------------
void
itk_warp_image_process
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
itk_warp_image_process
::_step()
{
  kwiver::vital::image_container_sptr image, size_image;

  ThermalImageType::SizeType output_size;
  WarpedThermalImageType::Pointer output_image;

  image = grab_from_port_using_trait( image );

  if( has_input_port_edge_using_trait( size_image ) )
  {
    size_image = grab_from_port_using_trait( size_image );

    output_size[0] = size_image->width();
    output_size[1] = size_image->height();
  }
  else
  {
    output_size[0] = image->width();
    output_size[1] = image->height();
  }

  auto itk_input_image = 
    ::itk::OpenCVImageBridge::CVMatToITKImage< viame::itk::ThermalImageType >(
      kwiver::arrows::ocv::image_container::vital_to_ocv(
        image->get_image(),
        kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  WarpThermalToOpticalImage(
    *itk_input_image,
    *d->m_transformation,
    output_size,
    output_image );

  push_to_port_using_trait( image,
    kwiver::vital::image_container_sptr(
      new kwiver::arrows::ocv::image_container(
      ::itk::OpenCVImageBridge::ITKImageToCVMat<
        viame::itk::WarpedThermalImageType >( output_image ),
      kwiver::arrows::ocv::image_container::BGR_COLOR ) ) );
}


// -----------------------------------------------------------------------------
void
itk_warp_image_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( size_image, optional );

  // -- output --
  declare_output_port_using_trait( image, optional );
}


// -----------------------------------------------------------------------------
void
itk_warp_image_process
::make_config()
{
  declare_config_using_trait( transformation_file );
}


// =============================================================================
itk_warp_image_process::priv
::priv()
  : m_transformation_file( "" )
{
}


itk_warp_image_process::priv
::~priv()
{
}

} // end namespace itk

} // end namespace viame
