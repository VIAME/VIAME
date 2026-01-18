#include "auto_detect_transform.h"

#include <vital/types/homography.h>

#include <fstream>

namespace viame
{

namespace kv = kwiver::vital;
namespace kva = kv::algo;

kv::transform_2d_sptr
auto_detect_transform_io
::load_( std::string const& filename ) const
{
  kv::transform_2d_sptr output;

  std::string::size_type idx = filename.rfind( '.' );
  std::string extension;

  if( idx != std::string::npos )
  {
    extension = filename.substr( idx + 1 );
  }

  if( extension == "h5" )
  {
    auto config = kv::config_block::empty_config();
    config->set_value( "transform_reader:type", "itk" );

    kva::transform_2d_io_sptr ti;

    kv::set_nested_algo_configuration<kva::transform_2d_io>(
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
      kv::homography_<double>::matrix_t homog;

      for( unsigned i = 0; i < 9; i++ )
      {
        homog( i / 3, i % 3 ) = parsed_file[ i + parsed_file.size() - 9 ];
      }

      output = std::make_shared< kv::homography_<double> >( homog );
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
::save_( std::string const& /*filename*/, kv::transform_2d_sptr /*data*/ ) const
{
}

} // end namespace viame
