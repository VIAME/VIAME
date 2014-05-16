/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "kw_archive_writer_process.h"

#include <types/maptk.h>
#include <types/kwiver.h>

#include <maptk/core/image_container.h>
#include <maptk/core/image.h>
#include <maptk/core/homography.h>

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

#include <vnl/io/vnl_io_matrix_fixed.h>
#include <vnl/io/vnl_io_vector_fixed.h>
#include <vil/io/vil_io_image_view.h>

#include <vnl/vnl_double_2.h>

namespace kwiver
{

  // This should be collected into kwiver types.h
  // define canonical type name for a set of corner points. (ul, ur, lr, ll) (lat, lon)
  sprokit::process::type_t const kw_archive_writer_process::kwiver_corner_points( "kwiver:corner_points" ); // or "kwiver:corner_points_ul_ur_lr_ll"

//----------------------------------------------------------------
// Private implementation class
class kw_archive_writer_process::priv
{
public:
  priv();
  ~priv();

  typedef float gsd_t;

  void write_frame_data(vsl_b_ostream& stream,
                        bool write_image,
                        int64_t time,
                        corner_points_t const& corners,
                        maptk::image const& img,
                        maptk::f2f_homography const& homog,
                        double gsd);

  static sprokit::process::port_t const port_timestamp;
  static sprokit::process::port_t const port_image;
  static sprokit::process::port_t const port_src_to_ref_homography;
  static sprokit::process::port_t const port_corner_points;
  static sprokit::process::port_t const port_gsd;


  // Configuration values
  std::string m_output_directory;
  static sprokit::config::key_t const config_output_directory;
  static sprokit::config::value_t const default_output_directory;

  std::string m_base_filename;
  static sprokit::config::key_t const config_base_filename;
  static sprokit::config::value_t const default_base_filename;

  bool m_separate_meta;
  static sprokit::config::key_t const config_separate_meta;
  static sprokit::config::value_t const default_separate_meta;

  std::string m_mission_id;
  static sprokit::config::key_t const config_mission_id;
  static sprokit::config::value_t const default_mission_id;

  std::string m_stream_id;
  static sprokit::config::key_t const config_stream_id;
  static sprokit::config::value_t const default_stream_id;

  bool m_compress_image;
  static sprokit::config::key_t const config_compress_image;
  static sprokit::config::value_t const default_compress_image;

  std::ofstream* m_index_stream;
  std::ofstream* m_meta_stream;
  vsl_b_ostream* m_meta_bstream;
  std::ofstream* m_data_stream;
  vsl_b_ostream* m_data_bstream;

  int m_data_version;
  std::vector < char > m_image_write_cache;

}; // end priv class


#define priv_t kw_archive_writer_process::priv

// -- config --
sprokit::config::key_t const priv_t::config_output_directory = sprokit::config::key_t( "output_directory" );
sprokit::config::value_t const priv_t::default_output_directory = sprokit::config::value_t( "." );

sprokit::config::key_t const priv_t::config_base_filename = sprokit::config::key_t( "base_filename" );
sprokit::config::value_t const priv_t::default_base_filename = sprokit::config::value_t( "kw_archive" );

sprokit::config::key_t const priv_t::config_separate_meta = sprokit::config::key_t( "separate_meta" );
sprokit::config::value_t const priv_t::default_separate_meta = sprokit::config::value_t( "true" );

sprokit::config::key_t const priv_t::config_mission_id = sprokit::config::key_t( "mission_id" );
sprokit::config::value_t const priv_t::default_mission_id = sprokit::config::value_t( "" );

sprokit::config::key_t const priv_t::config_stream_id = sprokit::config::key_t( "stream_id" );
sprokit::config::value_t const priv_t::default_stream_id = sprokit::config::value_t( "" );

sprokit::config::key_t const priv_t::config_compress_image = sprokit::config::key_t( "compress_image" );
sprokit::config::value_t const priv_t::default_compress_image = sprokit::config::value_t( "true" );

// -- ports --
  sprokit::process::port_t const priv_t::port_timestamp = sprokit::process::port_t("timestamp");
sprokit::process::port_t const priv_t::port_image = sprokit::process::port_t("image");
sprokit::process::port_t const priv_t::port_src_to_ref_homography = sprokit::process::port_t("src_to_ref_homography");
sprokit::process::port_t const priv_t::port_corner_points = sprokit::process::port_t("corner_points");
sprokit::process::port_t const priv_t::port_gsd = sprokit::process::port_t("gsd");


// ================================================================

kw_archive_writer_process
::kw_archive_writer_process( sprokit::config_t const& config )
  : process(config),
    d( new priv_t )
{
  make_ports();
  make_config();
}


kw_archive_writer_process
::~kw_archive_writer_process( )
{
}


// ----------------------------------------------------------------
void
kw_archive_writer_process
::_configure()
{
  // Examine the configuration
  d->m_output_directory = config_value< std::string > ( priv_t::config_output_directory );
  d->m_base_filename    = config_value< std::string > ( priv_t::config_base_filename );
  d->m_separate_meta    = config_value< bool > ( priv_t::config_separate_meta );
  d->m_mission_id       = config_value< std::string > ( priv_t::config_mission_id );
  d->m_stream_id        = config_value< std::string > ( priv_t::config_mission_id );
  d->m_compress_image   = config_value< bool > ( priv_t::config_compress_image );

  sprokit::process::_configure();
}


// ----------------------------------------------------------------
// Post connection initialization
void kw_archive_writer_process
::_init()
{
  std::string path = d->m_output_directory + "/" + d->m_base_filename;

  std::string index_filename = path + ".index";
  std::string meta_filename  = path + ".meta";
  std::string data_filename  = path + ".data";

  d->m_index_stream = new std::ofstream( index_filename.c_str(),
                                         std::ios::out | std::ios::trunc );

  if ( d->m_separate_meta )
  {
    // open metadata stream
    d->m_meta_stream = new std::ofstream( index_filename.c_str(),
                std::ios::out | std::ios::trunc | std::ios::binary );

    d->m_meta_bstream = new vsl_b_ostream( d->m_meta_stream );
    if ( ! *d->m_meta_stream )
    {
      std::string const reason = "Failed to open " + meta_filename + " for writing";
      throw sprokit::invalid_configuration_exception( name(), reason );
    }
  }

  d->m_data_stream = new std::ofstream( data_filename.c_str(),
             std::ios::out | std::ios::trunc | std::ios::binary );
  d->m_data_bstream = new vsl_b_ostream( d->m_data_stream );
  if ( ! *d->m_data_stream )
  {
    std::string const reason = "Failed to open " + data_filename + " for writing";
    throw sprokit::invalid_configuration_exception( name(), reason );
  }

  // Write file headers
  *d->m_index_stream
    << "4\n" // Version number
    << vul_file::basename( data_filename ) << "\n";

  if ( d->m_data_bstream != NULL )
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
  if ( d->m_compress_image )
  {
    d->m_data_version = 3;
  }
  else
  {
    d->m_data_version = 2;
  }

  vsl_b_write( *d->m_data_bstream, d->m_data_version ); // version number

  if ( d->m_meta_bstream )
  {
    vsl_b_write( *d->m_meta_bstream, static_cast< int > ( 2 ) ); // version number
  }

  if ( ! *d->m_index_stream || ! *d->m_data_stream ||
       ( d->m_separate_meta && ! *d->m_meta_stream ) )
  {
    static std::string const reason = "Failed while writing file headers";
    throw sprokit::invalid_configuration_exception( name(), reason );
  }

  process::_init();
} // kw_archive_writer_process::_init


// ----------------------------------------------------------------
void
kw_archive_writer_process
::_step()
{

  // timestamp
  // TBD
  int64_t time;

  // image
  maptk::image_container* img = grab_from_port_as< maptk::image_container* > ( priv::port_image );
  maptk::image image= img->get_image();

  // homography
  maptk::f2f_homography homog = grab_from_port_as< maptk::f2f_homography > ( priv::port_src_to_ref_homography );

  // corners
  corner_points_t corners = grab_from_port_as< corner_points_t > ( priv::port_corner_points );

  // gsd
  priv::gsd_t gsd = grab_from_port_as< priv::gsd_t > ( priv::port_gsd );


  *d->m_index_stream
    << time << " "
    << static_cast< int64_t > ( d->m_data_stream->tellp() )
    << "\n";

  d->write_frame_data( *d->m_data_bstream,
                       /*write image=*/ true,
                       time, corners, image, homog, gsd );
  if ( ! d->m_data_stream )
  {
    // throw ( ); //+ need general runtime exception
    // LOG_DEBUG("Failed while writing to .data stream");
  }

  if ( d->m_meta_bstream )
  {
    d->write_frame_data( *d->m_meta_bstream,
                         /*write image=*/ false,
                         time, corners, image, homog, gsd );
    if ( ! d->m_meta_stream )
    {
      // throw ( );
      // LOG_DEBUG("Failed while writing to .meta stream");
    }
  }

  sprokit::process::_step();
}


// ----------------------------------------------------------------
void
kw_archive_writer_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // declare input ports
  declare_input_port(
    priv::port_timestamp,
    kwiver_int_64,
    required,
    port_description_t( "Timestamp (frame num, time) for input image." ) );

  declare_input_port(
    priv::port_image,
    maptk_image,
    required,
    port_description_t( "Single frame image." ) );

  declare_input_port(
    priv::port_src_to_ref_homography,
    maptk_f2f_homography,
    required,
    port_description_t( "Source image to ref image homography" ) );

  declare_input_port(
    priv::port_corner_points,
    kwiver_corner_points,
    sprokit::process::port_flags_t(),
    port_description_t( "Four corner points for image in lat/lon units." ) );

  declare_input_port(
    priv::port_gsd,
    kwiver_float,
    required,
    port_description_t( "GSD for image in meters per pixel." ) );
}


// ----------------------------------------------------------------
void kw_archive_writer_process
::make_config()
{

  declare_configuration_key(
    priv::config_output_directory,
    priv::default_output_directory,
    sprokit::config::description_t( "Output directory where KWA will be written" ));

  declare_configuration_key(
    priv::config_base_filename,
    priv::default_base_filename,
    sprokit::config::description_t( "Base file name (no extension) for KWA component files" ));

  declare_configuration_key(
    priv::config_separate_meta,
    priv::default_separate_meta,
    sprokit::config::description_t( "Whether to write separate .meta file" ));

  declare_configuration_key(
    priv::config_mission_id,
    priv::default_mission_id,
    sprokit::config::description_t( "Mission ID to store in archive" ));

  declare_configuration_key(
    priv::config_stream_id,
    priv::default_stream_id,
    sprokit::config::description_t( "Stream ID to store in archive" ));

  declare_configuration_key(
    priv::config_compress_image,
    priv::default_compress_image,
    sprokit::config::description_t( "Whether to compress image data stored in archive" ));
}


// ----------------------------------------------------------------
void
priv_t
::write_frame_data(vsl_b_ostream& stream,
                   bool write_image,
                   int64_t time, // vidtk::timestamp const& ts,
                   kw_archive_writer_process::corner_points_t const& corner_pts,
                   maptk::image const& img,
                   maptk::f2f_homography const& s2r_homog,
                   double gsd)
{
  int64_t u_seconds = static_cast< int64_t > ( 0 ); // ts.time() );
  int64_t frame_num = static_cast< int64_t > ( s2r_homog.from_id() );
  int64_t ref_frame_num = static_cast< int64_t > ( s2r_homog.to_id() );

  // convert image in place
  vil_image_view < vxl_byte > image( img.first_pixel(),
                                     img.width(), // n_i
                                     img.height(), // n_j
                                     img.depth(), // n_planes
                                     img.w_step(), // i_step
                                     img.h_step(), // j_step
                                     img.d_step() // plane_step
    );

  // convert homography
  maptk::homography const& matrix( s2r_homog );  // upcast to base matrix
  vnl_matrix_fixed< double, 3, 3 > homog;

  // Copy matrix into vnl format
  for ( int x = 0; x < 2; ++x )
  {
    for ( int y = 0; y < 2; ++y )
    {
      homog( x, y ) = matrix( x, y );
    }
  }

  std::vector< vnl_vector_fixed< double, 2 > > corners;
  corners.push_back( vnl_double_2( corner_pts[0][1], corner_pts[0][0] ) ); // ul
  corners.push_back( vnl_double_2( corner_pts[1][1], corner_pts[1][0] ) ); // ur
  corners.push_back( vnl_double_2( corner_pts[2][1], corner_pts[2][0] ) ); // lr
  corners.push_back( vnl_double_2( corner_pts[3][1], corner_pts[3][0] ) ); // ll

  stream.clear_serialisation_records();
  vsl_b_write( stream, u_seconds );

  if ( write_image )
  {
    if ( this->m_data_version == 3 )
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
    else if ( this->m_data_version == 2 )
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
  vsl_b_write( stream, static_cast< int64_t > ( image.ni() ) );
  vsl_b_write( stream, static_cast< int64_t > ( image.nj() ) );

}

// ================================================================
kw_archive_writer_process::priv
::priv()
{
}


kw_archive_writer_process::priv
::~priv()
{
  // Must set pointers to zero to prevent multiple calls from doing
  // bad things.
  delete m_index_stream;
  m_index_stream = 0;

  delete m_meta_bstream;
  m_meta_bstream = 0;

  delete m_meta_stream;
  m_meta_stream = 0;

  delete m_data_bstream;
  m_data_bstream = 0;

  delete m_data_stream;
  m_data_stream = 0;
}


} // end namespace kwiver
