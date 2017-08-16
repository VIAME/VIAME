/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief KW Archive writer process implementation.
 */

#include "kw_archive_writer_process.h"

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/homography_f2f.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

#include <fstream>
#include <vector>
#include <stdint.h>

#include <vul/vul_file.h>
#include <vsl/vsl_binary_io.h>
#include <vsl/vsl_vector_io.h>

#include <vil/vil_image_view.h>
#include <vil/vil_pixel_format.h>
#include <vil/vil_stream_core.h>
#include <vil/file_formats/vil_jpeg.h>
#include <vil/io/vil_io_image_view.h>

#include <vnl/io/vnl_io_matrix_fixed.h>
#include <vnl/io/vnl_io_vector_fixed.h>

#include <vnl/vnl_double_2.h>

// instantiate vsl vector routine
#include <vsl/vsl_vector_io.hxx>
VSL_VECTOR_IO_INSTANTIATE( char );


namespace kwiver
{

// -- Configuration Items --
create_config_trait( output_directory, std::string,
  ".", "Output directory where KWA will be written" );
create_config_trait( base_filename, std::string,
  "", "Base file name (no extension) for KWA component files" );
create_config_trait( separate_meta, bool,
  "true", "Whether to write separate .meta file" );
create_config_trait( mission_id, std::string,
  "", "Mission ID to store in archive" );
create_config_trait( stream_id, std::string,
  "", "Stream ID to store in archive" );
create_config_trait( compress_image, bool,
  "true", "Whether to compress image data stored in archive" );

create_type_trait( bool,
  "kwiver:bool", bool );
create_port_trait( filename, file_name,
  "KWA input filename" );
create_port_trait( stream_id, string,
  "Stream ID to place in file" );
create_port_trait( complete_flag, bool,
  "KWA complete flag" );


//-----------------------------------------------------------------------------------
// Private implementation class
class kw_archive_writer_process::priv
{
public:
  priv( kw_archive_writer_process* parent );
  ~priv();

  void write_frame_data( vsl_b_ostream& stream,
                         bool write_image,
                         kwiver::vital::timestamp const& time,
                         kwiver::vital::geo_corner_points const& corners,
                         kwiver::vital::image const& img,
                         kwiver::vital::f2f_homography const& homog,
                         kwiver::vital::gsd_t gsd );

  static sprokit::process::port_t const port_timestamp;
  static sprokit::process::port_t const port_image;
  static sprokit::process::port_t const port_src_to_ref_homography;
  static sprokit::process::port_t const port_corner_points;
  static sprokit::process::port_t const port_gsd;
  static sprokit::process::port_t const port_stream_id;
  static sprokit::process::port_t const port_filename;

  kw_archive_writer_process* m_parent;

  // Configuration values
  std::string m_output_directory;
  std::string m_base_filename;
  bool m_separate_meta;
  std::string m_mission_id;
  std::string m_stream_id;
  bool m_compress_image;

  // local storage
  std::unique_ptr< std::ofstream > m_index_stream;
  std::unique_ptr< std::ofstream > m_meta_stream;
  std::unique_ptr< vsl_b_ostream > m_meta_bstream;
  std::unique_ptr< std::ofstream > m_data_stream;
  std::unique_ptr< vsl_b_ostream > m_data_bstream;

  int m_data_version;
  std::vector < char > m_image_write_cache;

}; // end priv class

#define priv_t kw_archive_writer_process::priv

// ==================================================================================

kw_archive_writer_process
::kw_archive_writer_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new kw_archive_writer_process::priv( this ) )
{
  attach_logger( kwiver::vital::get_logger( name() ) );

  make_ports();
  make_config();
}


kw_archive_writer_process
::~kw_archive_writer_process( )
{
}


//-----------------------------------------------------------------------------------
void
kw_archive_writer_process
::_configure()
{
  // Examine the configuration
  d->m_output_directory = config_value_using_trait( output_directory );
  d->m_base_filename    = config_value_using_trait( base_filename );
  d->m_separate_meta    = config_value_using_trait( separate_meta );
  d->m_mission_id       = config_value_using_trait( mission_id );
  d->m_stream_id        = config_value_using_trait( stream_id );
  d->m_compress_image   = config_value_using_trait( compress_image );
}


//-----------------------------------------------------------------------------------
// Post connection initialization
void
kw_archive_writer_process
::_init()
{
  if( d->m_base_filename.empty() )
  {
    return;
  }

  std::string path = d->m_output_directory + "/" + d->m_base_filename;

  // Make sure directory exists
  vul_file::make_directory_path( d->m_output_directory );

  std::string index_filename = path + ".index";
  std::string meta_filename  = path + ".meta";
  std::string data_filename  = path + ".data";

  d->m_index_stream.reset( new std::ofstream( index_filename.c_str(),
                                std::ios::out | std::ios::trunc ) );
  if( ! *d->m_index_stream )
  {
    std::string const reason = "Failed to open " + index_filename + " for writing";
    throw sprokit::invalid_configuration_exception( name(), reason );
  }

  if( d->m_separate_meta )
  {
    // open metadata stream
    d->m_meta_stream.reset( new std::ofstream( meta_filename.c_str(),
             std::ios::out | std::ios::trunc | std::ios::binary ) );

    if( ! *d->m_meta_stream )
    {
      std::string const reason = "Failed to open " + meta_filename + " for writing";
      throw sprokit::invalid_configuration_exception( name(), reason );
    }

    d->m_meta_bstream.reset( new vsl_b_ostream( d->m_meta_stream.get() ) );
  }

  d->m_data_stream.reset( new std::ofstream( data_filename.c_str(),
             std::ios::out | std::ios::trunc | std::ios::binary ) );

  d->m_data_bstream.reset( new vsl_b_ostream( d->m_data_stream.get() ) );

  if( ! *d->m_data_stream )
  {
    std::string const reason = "Failed to open " + data_filename + " for writing";
    throw sprokit::invalid_configuration_exception( name(), reason );
  }

  // Write file headers
  *d->m_index_stream
    << "4\n" // Version number
    << vul_file::basename( data_filename ) << "\n";

  if( d->m_data_bstream != NULL )
  {
    *d->m_index_stream << vul_file::basename( meta_filename ) << "\n";
  }
  else
  {
    *d->m_index_stream << "\n";
  }

  *d->m_index_stream
    << d->m_mission_id << "\n"
    << d->m_stream_id << "\n";

  // version depends on compression option
  if( d->m_compress_image )
  {
    d->m_data_version = 3;
  }
  else
  {
    d->m_data_version = 2;
  }

  vsl_b_write( *d->m_data_bstream, d->m_data_version ); // version number

  if( d->m_meta_bstream )
  {
    vsl_b_write( *d->m_meta_bstream, static_cast< int > ( 2 ) ); // version number
  }

  if( ! *d->m_index_stream || ! *d->m_data_stream ||
      ( d->m_separate_meta && ! *d->m_meta_stream ) )
  {
    static std::string const reason = "Failed while writing file headers";
    throw sprokit::invalid_configuration_exception( name(), reason );
  }
} // kw_archive_writer_process::_init


//-----------------------------------------------------------------------------------
void
kw_archive_writer_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_input_using_trait( timestamp );

  // image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );

  // check for empty image
  if( !img || img->width() == 0 || img->height() == 0 )
  {
    push_to_port_using_trait( complete_flag, true );
    return;
  }

  kwiver::vital::image image = img->get_image();

  // homography
  kwiver::vital::f2f_homography homog( Eigen::Matrix< double, 3, 3 >(), -1, -1 );

  if( process::has_input_port_edge( "homography_src_to_ref" ) )
  {
    homog = grab_from_port_using_trait( homography_src_to_ref );
  }

  // corners
  kwiver::vital::geo_corner_points corners;

  if( process::has_input_port_edge( "corner_points" ) )
  {
    corners = grab_input_using_trait( corner_points );
  }

  // gsd
  kwiver::vital::gsd_t gsd = -1.0;

  if( process::has_input_port_edge( "gsd" ) )
  {
    gsd = grab_input_using_trait( gsd );
  }

  // filename
  kwiver::vital::string_t filename;

  if( process::has_input_port_edge( "filename" ) )
  {
    filename = grab_input_using_trait( filename );
  }

  // stream id
  kwiver::vital::string_t stream_id;

  if( process::has_input_port_edge( "stream_id" ) )
  {
    stream_id = grab_input_using_trait( stream_id );
  }

  // Check to see if filename or stream id updated
  if( !stream_id.empty() && d->m_stream_id != stream_id )
  {
    d->m_stream_id = stream_id;
  }

  if( !filename.empty() && d->m_base_filename != filename )
  {
    d->m_base_filename = filename;

    _init();
  }

  if( d->m_base_filename.empty() )
  {
    static std::string const reason = "No output filename specified";
    throw sprokit::invalid_configuration_exception( name(), reason );
  }

  // Beginning writing this frame to KWA
  LOG_DEBUG( logger(), "processing frame " << frame_time );

  *d->m_index_stream
    << static_cast< vxl_int_64 > ( frame_time.get_time_usec() ) << " " // in micro-seconds
    << static_cast< int64_t > ( d->m_data_stream->tellp() )
    << std::endl;

  d->write_frame_data( *d->m_data_bstream,
                       /*write image=*/ true,
                       frame_time, corners, image, homog, gsd );

  if( ! d->m_data_stream )
  {
    // throw ( ); //+ need general runtime exception
    // LOG_DEBUG("Failed while writing to .data stream");
  }

  if( d->m_meta_bstream )
  {
    d->write_frame_data( *d->m_meta_bstream,
                         /*write48 image=*/ false,
                         frame_time, corners, image, homog, gsd );

    if( ! d->m_meta_stream )
    {
      // throw ( );
      // LOG_DEBUG("Failed while writing to .meta stream");
    }
  }

  d->m_meta_stream->flush();
  d->m_data_stream->flush();

  push_to_port_using_trait( complete_flag, true );
} // kw_archive_writer_process::_step


//-----------------------------------------------------------------------------------
void
kw_archive_writer_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t opt_static;
  opt_static.insert( flag_input_static );

  // declare input ports
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( homography_src_to_ref, optional );
  declare_input_port_using_trait( corner_points, opt_static );
  declare_input_port_using_trait( gsd, opt_static );
  declare_input_port_using_trait( filename, opt_static );
  declare_input_port_using_trait( stream_id, opt_static );

  declare_output_port_using_trait( complete_flag, opt_static );
}


//-----------------------------------------------------------------------------------
void
kw_archive_writer_process
::make_config()
{
  declare_config_using_trait( output_directory );
  declare_config_using_trait( base_filename );
  declare_config_using_trait( separate_meta );
  declare_config_using_trait( mission_id );
  declare_config_using_trait( stream_id );
  declare_config_using_trait( compress_image );
}


//-----------------------------------------------------------------------------------
void
priv_t
::write_frame_data( vsl_b_ostream& stream,
                    bool write_image,
                    kwiver::vital::timestamp const& time,
                    kwiver::vital::geo_corner_points const& corner_pts,
                    kwiver::vital::image const& img,
                    kwiver::vital::f2f_homography const& s2r_homog,
                    double gsd )
{
  vxl_int_64 u_seconds = static_cast< vxl_int_64 > ( time.get_time_usec() );
  vxl_int_64 frame_num = static_cast< vxl_int_64 > ( time.get_frame() );
  vxl_int_64 ref_frame_num = static_cast< vxl_int_64 > ( s2r_homog.to_id() );

  // Validate expected image type
  auto trait = img.pixel_traits();
  if( trait.type != kwiver::vital::image_pixel_traits::UNSIGNED || trait.num_bytes != 1 )
  {
    LOG_ERROR( m_parent->logger(), "Input image type is not of unsigned char pixel type" );
    return;
  }

  // convert image in place
  vil_image_view < vxl_byte > image( static_cast< const vxl_byte* >( img.first_pixel() ),
                                     img.width(), // n_i
                                     img.height(), // n_j
                                     img.depth(), // n_planes
                                     img.w_step(), // i_step
                                     img.h_step(), // j_step
                                     img.d_step() // plane_step
    );

  // convert homography
  Eigen::Matrix< double, 3, 3 > matrix= s2r_homog.homography()->matrix();
  vnl_matrix_fixed< double, 3, 3 > homog;

  // Copy matrix into vnl format
  for( int x = 0; x < 3; ++x )
  {
    for( int y = 0; y < 3; ++y )
    {
      homog( x, y ) = matrix( x, y );
    }
  }

  std::vector< vnl_vector_fixed< double, 2 > > corners; // (x,y)
  corners.push_back( vnl_double_2( corner_pts.p1.longitude(), corner_pts.p1.latitude() ) ); // ul
  corners.push_back( vnl_double_2( corner_pts.p2.longitude(), corner_pts.p2.latitude() ) ); // ur
  corners.push_back( vnl_double_2( corner_pts.p3.longitude(), corner_pts.p3.latitude() ) ); // lr
  corners.push_back( vnl_double_2( corner_pts.p4.longitude(), corner_pts.p4.latitude() ) ); // ll

  stream.clear_serialisation_records();
  vsl_b_write( stream, u_seconds );

  if( write_image )
  {
    if( this->m_data_version == 3 )
    {
      vsl_b_write( stream, 'J' ); // J=jpeg
      vil_stream* mem_stream = new vil_stream_core();
      mem_stream->ref();
      vil_jpeg_file_format fmt;
      vil_image_resource_sptr img_res =
        fmt.make_output_image( mem_stream,
                               image.ni(), image.nj(), image.nplanes(),
                               VIL_PIXEL_FORMAT_BYTE );
      img_res->put_view( image );
      this->m_image_write_cache.resize( mem_stream->file_size() );
      mem_stream->seek( 0 );
      // LOG_DEBUG( "Compressed image is " << mem_stream->file_size() << " bytes" );
      mem_stream->read( &this->m_image_write_cache[0], mem_stream->file_size() );
      vsl_b_write( stream, this->m_image_write_cache );
      mem_stream->unref(); // allow for automatic delete
    }
    else if( this->m_data_version == 2 )
    {
      vsl_b_write( stream, image );
    }
    else
    {
      // throw (); unexpected version number
    }
  }

  vsl_b_write( stream, homog );
  vsl_b_write( stream, corners );
  vsl_b_write( stream, gsd );
  vsl_b_write( stream, frame_num );
  vsl_b_write( stream, ref_frame_num );
  vsl_b_write( stream, static_cast< vxl_int_64 > ( image.ni() ) );
  vsl_b_write( stream, static_cast< vxl_int_64 > ( image.nj() ) );
}

// ==================================================================================
kw_archive_writer_process::priv
::priv(kw_archive_writer_process* parent)
  : m_parent( parent )
{
}


kw_archive_writer_process::priv
::~priv()
{
}


} // end namespace kwiver
