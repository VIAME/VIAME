/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

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

  if( !WarpThermalToOpticalImage(
        *itk_input_image,
        *d->m_transformation,
        output_size,
        output_image ) )
  {
    push_to_port_using_trait( image, kwiver::vital::image_container_sptr() );
    return;
  }

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
