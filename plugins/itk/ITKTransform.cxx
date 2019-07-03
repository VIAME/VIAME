
#include "ITKTransform.h"

#include <itkTransformFileReader.h>


namespace viame
{

namespace itk
{

ITKTransform
::ITKTransform( viame::itk::NetTransformType::Pointer transform )
{
  m_transform = transform;
}

ITKTransform
::~ITKTransform()
{
}

kwiver::vital::transform_2d_sptr
ITKTransform
::clone() const
{
  return kwiver::vital::transform_2d_sptr( new ITKTransform( m_transform ) );
}

kwiver::vital::vector_2d
ITKTransform
::map( kwiver::vital::vector_2d const& p ) const
{
  TransformFloatType input[2] = { p[0], p[1] };

  NetTransformType::OutputPointType output =
    m_transform->TransformPoint(
      NetTransformType::InputPointType( input ) );

  return kwiver::vital::vector_2d( output[0], output[1] );
}

ITKTransformIO
::ITKTransformIO()
{
}

ITKTransformIO
::~ITKTransformIO()
{
}


kwiver::vital::config_block_sptr
ITKTransformIO
::get_configuration() const
{
  return kwiver::vital::algo::transform_2d_io::get_configuration();
}

void
ITKTransformIO
::set_configuration( kwiver::vital::config_block_sptr /*config*/ )
{
  return;
}

bool
ITKTransformIO
::check_configuration( kwiver::vital::config_block_sptr /*config*/ ) const
{
  return true;
}

kwiver::vital::transform_2d_sptr
ITKTransformIO
::load_( std::string const& filename ) const
{
  ::itk::TransformFileReaderTemplate< TransformFloatType >::Pointer reader =
    ::itk::TransformFileReaderTemplate< TransformFloatType >::New();

  reader->SetFileName( filename );
  reader->Update();

  if( reader->GetTransformList()->size() != 1 )
  {
    throw std::runtime_error( "Unable to load: " + filename );
  }

  return kwiver::vital::transform_2d_sptr(
    new viame::itk::ITKTransform(
      static_cast< NetTransformType* >(
        reader->GetTransformList()->begin()->GetPointer() ) ) );
}

void
ITKTransformIO
::save_( std::string const& /*filename*/, kwiver::vital::transform_2d_sptr /*data*/ ) const
{
}

} // end namespace itk

} // end namespace viame
