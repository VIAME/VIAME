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
 * \brief Register images using ITK.
 */

#include "RegistrationProcess.h"
#include "RegisterOpticalAndThermal.h"

#include <itkOpenCVImageBridge.h>

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/homography.h>

#include <arrows/ocv/image_container.h>

using namespace viame::core;

namespace viame
{

namespace itk
{

// =============================================================================

itk_eo_ir_registration_process
::itk_eo_ir_registration_process( kwiver::vital::config_block_sptr const& config )
  : align_multimodal_imagery_process( config )
{
}


itk_eo_ir_registration_process
::~itk_eo_ir_registration_process()
{
}

void
itk_eo_ir_registration_process
::attempt_registration( const buffered_frame& optical,
                        const buffered_frame& thermal,
                        const bool optical_dom )
{
  viame::itk::NetTransformType::Pointer output_transform;

  auto itk_optical_image = 
    ::itk::OpenCVImageBridge::CVMatToITKImage< viame::itk::OpticalImageType >(
      kwiver::arrows::ocv::image_container::vital_to_ocv(
        optical.image->get_image(),
        kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  auto itk_thermal_image = 
    ::itk::OpenCVImageBridge::CVMatToITKImage< viame::itk::ThermalImageType >(
      kwiver::arrows::ocv::image_container::vital_to_ocv(
        thermal.image->get_image(),
        kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  if( PerformRegistration( *itk_optical_image, *itk_thermal_image, output_transform ) )
  {
    // Convert matrix to kwiver
    kwiver::vital::homography_sptr optical_to_thermal(
      new kwiver::vital::homography_< double >() );

    kwiver::vital::matrix_3x3d& net_output =
      dynamic_cast< kwiver::vital::homography_< double >* >(
        optical_to_thermal.get() )->get_matrix();

    net_output = kwiver::vital::matrix_3x3d::Identity();

    for( unsigned n = 0; n < output_transform->GetNumberOfTransforms(); n++ )
    {
      viame::itk::AffineTransformType* itk_affine_transform = 
        dynamic_cast< viame::itk::AffineTransformType* >(
          output_transform->GetNthTransform( n ).GetPointer() );

      if( !itk_affine_transform )
      {
        throw std::runtime_error( "Unknown transformation type received" );
      }

      const auto& in_values = itk_affine_transform->GetMatrix();
      kwiver::vital::matrix_3x3d next_homog;

      for( unsigned r = 0; r < viame::itk::Dimension; ++r )
      {
        for( unsigned c = 0; c < viame::itk::Dimension; ++c )
        {
          next_homog( r, c ) = in_values( r, c );
        }
      }

      if( viame::itk::Dimension == 2 )
      {
        next_homog( 2, 0 ) = 0;
        next_homog( 2, 1 ) = 0;
        next_homog( 2, 2 ) = 1;

        next_homog( 1, 2 ) = itk_affine_transform->GetOffset()[ 1 ];
        next_homog( 0, 2 ) = itk_affine_transform->GetOffset()[ 0 ];
      }

      net_output = net_output * next_homog;
    }

    // Output required elements depending on connections
    push_to_port_using_trait( optical_image, optical.image );
    push_to_port_using_trait( optical_file_name, optical.name );
    push_to_port_using_trait( thermal_image, thermal.image );
    push_to_port_using_trait( thermal_file_name, thermal.name );
    push_to_port_using_trait( timestamp, ( optical_dom ? optical.ts : thermal.ts ) );
    push_to_port_using_trait( success_flag, true );
    push_to_port_using_trait( optical_to_thermal_homog, optical_to_thermal );

    if( count_output_port_edges_using_trait( thermal_to_optical_homog ) > 0 )
    {
      push_to_port_using_trait( thermal_to_optical_homog,
        std::static_pointer_cast< kwiver::vital::homography >(
          optical_to_thermal->inverse() ) );
    }

    // Warp image if required
    if( count_output_port_edges_using_trait( warped_thermal_image ) > 0 )
    {
      WarpedThermalImageType::Pointer warped_image;

      if( WarpThermalToOpticalImage(
        *itk_optical_image, *itk_thermal_image, *output_transform, warped_image ) )
      {
        push_to_port_using_trait( warped_thermal_image,
          kwiver::vital::image_container_sptr(
            new kwiver::arrows::ocv::image_container(
            ::itk::OpenCVImageBridge::ITKImageToCVMat<
              viame::itk::WarpedThermalImageType >( warped_image ),
            kwiver::arrows::ocv::image_container::BGR_COLOR ) ) );
      }
      else
      {
        push_to_port_using_trait( warped_thermal_image,
          kwiver::vital::image_container_sptr() );
      }
    }

    if( count_output_port_edges_using_trait( warped_optical_image ) > 0 )
    {
      WarpedOpticalImageType::Pointer warped_image;

      if( WarpOpticalToThermalImage(
        *itk_optical_image, *itk_thermal_image, *output_transform, warped_image ) )
      {
        push_to_port_using_trait( warped_optical_image,
          kwiver::vital::image_container_sptr(
            new kwiver::arrows::ocv::image_container(
            ::itk::OpenCVImageBridge::ITKImageToCVMat<
              viame::itk::WarpedOpticalImageType >( warped_image ),
            kwiver::arrows::ocv::image_container::BGR_COLOR ) ) );
      }
      else
      {
        push_to_port_using_trait( warped_optical_image,
          kwiver::vital::image_container_sptr() );
      }
    }
  }
  else
  {
    if( optical_dom )
    {
      output_no_match( optical, 0 );
    }
    else
    {
      output_no_match( thermal, 1 );
    }
  }
}

} // end namespace itk

} // end namespace viame
