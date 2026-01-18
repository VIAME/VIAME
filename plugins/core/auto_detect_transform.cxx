#include "auto_detect_transform.h"

#include <vital/algo/algorithm.txx>

#include <vital/types/homography.h>

#include <fstream>

namespace viame
{


auto_detect_transform_io
::auto_detect_transform_io()
{
}

auto_detect_transform_io
::~auto_detect_transform_io()
{
}


kwiver::vital::config_block_sptr
auto_detect_transform_io
::get_configuration() const
{
  return kwiver::vital::algo::transform_2d_io::get_configuration();
}

void
auto_detect_transform_io
::set_configuration( kwiver::vital::config_block_sptr /*config*/ )
{
  return;
}

bool
auto_detect_transform_io
::check_configuration( kwiver::vital::config_block_sptr /*config*/ ) const
{
  return true;
}

kwiver::vital::transform_2d_sptr
auto_detect_transform_io
::load_( std::string const& filename ) const
{
  kwiver::vital::transform_2d_sptr output;

  std::string::size_type idx = filename.rfind( '.' );
  std::string extension;

  if( idx != std::string::npos )
  {
    extension = filename.substr( idx + 1 );
  }

  if( extension == "h5" )
  {
    auto config = kwiver::vital::config_block::empty_config();
    config->set_value( "transform_reader:type", "itk" );

    kwiver::vital::algo::transform_2d_io_sptr ti;

    kwiver::vital::set_nested_algo_configuration<kwiver::vital::algo::transform_2d_io>(
      "transform_reader", config, ti );
  
    if( ti )
    {
      output = ti->load( filename );
    }
  }
  else
  {
    std::vector< double > parsed_file;
    double value;

    std::ifstream input( filename );

    while( input >> value )
    {
      parsed_file.push_back( value );
    }

    input.close();

    if( parsed_file.size() >= 9 && parsed_file.size() <= 13 )
    {
      kwiver::vital::homography_<double>::matrix_t homog;

      for( unsigned i = 0; i < 9; i++ )
      {
        homog( i / 3, i % 3 ) = parsed_file[ i + parsed_file.size() - 9 ];
      }

      output = std::make_shared< kwiver::vital::homography_<double> >( homog );
    }
  }

  if( !output )
  {
    throw std::runtime_error( "Unable to decipher transform format" );
  }

  return output;
}

void
auto_detect_transform_io
::save_( std::string const& /*filename*/, kwiver::vital::transform_2d_sptr /*data*/ ) const
{
}

} // end namespace viame
