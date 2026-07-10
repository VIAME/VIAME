// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief Implementation of camera rig I/O functions

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <regex>
#include <cstdint>
#include <cstring>
#include <map>

#include <vital/exceptions.h>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/vector.hpp>

#include <kwiversys/SystemTools.hxx>

#ifdef VIAME_ENABLE_ZLIB
#include <zlib.h>
#include <cstring>
#include <map>
#endif

#include "camera_rig_io.h"
#include "camera_io.h"

namespace { // anon

// -----------------------------------------------------------------------------
// OpenCV YAML matrix parsing utilities
// OpenCV YAML format uses !!opencv-matrix tag with rows, cols, dt, and data fields
// -----------------------------------------------------------------------------

// Trim whitespace from both ends of a string
std::string trim( const std::string& str )
{
  size_t first = str.find_first_not_of( " \t\n\r" );
  if( first == std::string::npos )
  {
    return "";
  }
  size_t last = str.find_last_not_of( " \t\n\r" );
  return str.substr( first, last - first + 1 );
}

// Parse a comma-separated list of doubles from a string like "[ 1.0, 2.0, 3.0 ]"
std::vector<double> parse_double_array( const std::string& str )
{
  std::vector<double> values;
  std::string content = str;

  // Remove brackets
  size_t start = content.find( '[' );
  size_t end = content.rfind( ']' );
  if( start != std::string::npos && end != std::string::npos && end > start )
  {
    content = content.substr( start + 1, end - start - 1 );
  }

  // Split by comma and parse
  std::stringstream ss( content );
  std::string token;
  while( std::getline( ss, token, ',' ) )
  {
    std::string trimmed = trim( token );
    if( !trimmed.empty() )
    {
      try
      {
        values.push_back( std::stod( trimmed ) );
      }
      catch( ... )
      {
        // Skip invalid values
      }
    }
  }
  return values;
}

// Structure to hold parsed OpenCV matrix data
struct ocv_yaml_matrix
{
  int rows = 0;
  int cols = 0;
  std::string dtype;
  std::vector<double> data;

  bool is_valid() const
  {
    return rows > 0 && cols > 0 &&
           static_cast<size_t>(rows * cols) == data.size();
  }

  double at( int r, int c ) const
  {
    return data[ r * cols + c ];
  }
};

// Parse OpenCV YAML file and extract matrices by name
// Returns a map of matrix name -> ocv_yaml_matrix
std::map<std::string, ocv_yaml_matrix> parse_ocv_yaml_file( const std::string& filename )
{
  std::map<std::string, ocv_yaml_matrix> matrices;

  std::ifstream file( filename );
  if( !file.is_open() )
  {
    return matrices;
  }

  std::string line;
  std::string current_matrix_name;
  ocv_yaml_matrix current_matrix;
  bool in_matrix = false;
  std::string data_buffer;
  bool collecting_data = false;

  while( std::getline( file, line ) )
  {
    std::string trimmed = trim( line );

    // Skip YAML header and comments
    if( trimmed.empty() || trimmed[0] == '%' || trimmed[0] == '#' || trimmed == "---" )
    {
      continue;
    }

    // Check for new matrix definition (name followed by !!opencv-matrix)
    if( trimmed.find( "!!opencv-matrix" ) != std::string::npos )
    {
      // Save previous matrix if valid
      if( in_matrix && !current_matrix_name.empty() && current_matrix.is_valid() )
      {
        matrices[current_matrix_name] = current_matrix;
      }

      // Start new matrix
      size_t colon_pos = trimmed.find( ':' );
      if( colon_pos != std::string::npos )
      {
        current_matrix_name = trim( trimmed.substr( 0, colon_pos ) );
      }
      current_matrix = ocv_yaml_matrix();
      in_matrix = true;
      collecting_data = false;
      data_buffer.clear();
      continue;
    }

    if( !in_matrix )
    {
      continue;
    }

    // Parse matrix properties
    size_t colon_pos = trimmed.find( ':' );
    if( colon_pos != std::string::npos )
    {
      std::string key = trim( trimmed.substr( 0, colon_pos ) );
      std::string value = trim( trimmed.substr( colon_pos + 1 ) );

      if( key == "rows" )
      {
        current_matrix.rows = std::stoi( value );
      }
      else if( key == "cols" )
      {
        current_matrix.cols = std::stoi( value );
      }
      else if( key == "dt" )
      {
        current_matrix.dtype = value;
      }
      else if( key == "data" )
      {
        // Data might start on this line or continue on next lines
        collecting_data = true;
        if( !value.empty() )
        {
          data_buffer = value;
          // Check if data is complete (ends with ])
          if( value.find( ']' ) != std::string::npos )
          {
            current_matrix.data = parse_double_array( data_buffer );
            collecting_data = false;
          }
        }
      }
    }
    else if( collecting_data )
    {
      // Continue collecting data array
      data_buffer += " " + trimmed;
      if( trimmed.find( ']' ) != std::string::npos )
      {
        current_matrix.data = parse_double_array( data_buffer );
        collecting_data = false;
      }
    }
  }

  // Save last matrix if valid
  if( in_matrix && !current_matrix_name.empty() && current_matrix.is_valid() )
  {
    matrices[current_matrix_name] = current_matrix;
  }

  return matrices;
}

//
// helper class for left / right camera intrinsics shared beween json and yaml loaders
//
// (since this class has a default ctor, we can store it in a "left"/"right" map)
//

class intrinsics_builder
{
private:
  bool valid_;
  double fx_, fy_, cx_, cy_;
  Eigen::VectorXd dist_;

public:
  intrinsics_builder(): valid_(false) {}
  intrinsics_builder( double fx,                // focal point x
                      double fy,                // focal point y
                      double cx,                // principal point x
                      double cy,                // principal point y
                      const Eigen::VectorXd& dist ) // distortion parameters
    : valid_(true), fx_(fx), fy_(fy), cx_(cx), cy_(cy), dist_(dist)
  {}

  kwiver::vital::camera_intrinsics_sptr make_intrinsics( void ) const
  {
    if (!valid_)
    {
      throw std::logic_error("trying to build a camera from uninitialized intrinsics");
    }

    double focal_length = fx_;
    double dx = 2*cx_, dy = 2*cy_;
    kwiver::vital::vector_2d principal_point(cx_, cy_);
    auto const aspect_ratio = fx_ / fy_, skew = 0.0;

    return std::make_shared<kwiver::vital::simple_camera_intrinsics>(
      focal_length,
      principal_point,
      aspect_ratio,
      skew,
      dist_,
      dx, dy
      );
  }
};

#ifdef VIAME_ENABLE_ZLIB
// -----------------------------------------------------------------------------
// NPZ file reading utilities
// NPZ files are ZIP archives containing .npy files
// -----------------------------------------------------------------------------

#pragma pack(push, 1)
struct zip_local_file_header
{
  uint32_t signature;        // 0x04034b50
  uint16_t version_needed;
  uint16_t flags;
  uint16_t compression;
  uint16_t mod_time;
  uint16_t mod_date;
  uint32_t crc32;
  uint32_t compressed_size;
  uint32_t uncompressed_size;
  uint16_t filename_len;
  uint16_t extra_len;
};
#pragma pack(pop)

// Read a little-endian value from a byte buffer
template<typename T>
T read_le( const unsigned char* data )
{
  T value = 0;
  for( size_t i = 0; i < sizeof(T); ++i )
  {
    value |= static_cast<T>(data[i]) << (8 * i);
  }
  return value;
}

// Parse a NumPy array header to get shape and dtype info
// Returns true if successful, populates shape and is_fortran_order
bool parse_npy_header( const std::vector<unsigned char>& data,
                       std::vector<size_t>& shape,
                       bool& is_fortran_order,
                       char& dtype_char,
                       size_t& dtype_size,
                       size_t& header_size )
{
  // NPY format: magic string + version + header_len + header (Python dict as string)
  if( data.size() < 10 )
  {
    return false;
  }

  // Check magic number: 0x93NUMPY
  if( data[0] != 0x93 || data[1] != 'N' || data[2] != 'U' ||
      data[3] != 'M' || data[4] != 'P' || data[5] != 'Y' )
  {
    return false;
  }

  uint8_t major_version = data[6];
  // uint8_t minor_version = data[7];

  uint32_t header_len;
  size_t header_start;
  if( major_version == 1 )
  {
    header_len = read_le<uint16_t>( &data[8] );
    header_start = 10;
  }
  else
  {
    header_len = read_le<uint32_t>( &data[8] );
    header_start = 12;
  }

  if( data.size() < header_start + header_len )
  {
    return false;
  }

  header_size = header_start + header_len;

  // Parse the header string (simplified Python dict parser)
  std::string header( data.begin() + header_start, data.begin() + header_start + header_len );

  // Find 'fortran_order': False or True
  is_fortran_order = ( header.find("'fortran_order': True") != std::string::npos ||
                       header.find("'fortran_order':True") != std::string::npos );

  // Find 'descr': '<f8' or similar
  auto descr_pos = header.find("'descr':");
  if( descr_pos == std::string::npos )
  {
    return false;
  }

  auto quote_start = header.find("'", descr_pos + 8);
  if( quote_start == std::string::npos )
  {
    return false;
  }
  auto quote_end = header.find("'", quote_start + 1);
  if( quote_end == std::string::npos )
  {
    return false;
  }
  std::string descr = header.substr( quote_start + 1, quote_end - quote_start - 1 );

  // Parse dtype: e.g., "<f8" means little-endian float64
  if( descr.size() < 2 )
  {
    return false;
  }
  dtype_char = descr[1];  // 'f' for float, 'i' for int, etc.
  dtype_size = std::stoul( descr.substr(2) );

  // Find 'shape': (3, 3) or (3,) or ()
  auto shape_pos = header.find("'shape':");
  if( shape_pos == std::string::npos )
  {
    return false;
  }

  auto paren_start = header.find("(", shape_pos);
  auto paren_end = header.find(")", paren_start);
  if( paren_start == std::string::npos || paren_end == std::string::npos )
  {
    return false;
  }

  std::string shape_str = header.substr( paren_start + 1, paren_end - paren_start - 1 );
  shape.clear();

  // Parse comma-separated integers (handles Python 2 long literals like "3L")
  size_t pos = 0;
  while( pos < shape_str.size() )
  {
    // Skip whitespace and commas
    while( pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',') )
    {
      ++pos;
    }
    if( pos >= shape_str.size() )
    {
      break;
    }

    // Read number
    size_t num_start = pos;
    while( pos < shape_str.size() && shape_str[pos] >= '0' && shape_str[pos] <= '9' )
    {
      ++pos;
    }
    if( pos > num_start )
    {
      shape.push_back( std::stoul( shape_str.substr(num_start, pos - num_start) ) );
    }

    // Skip any trailing characters after number (e.g., 'L' from Python 2 long literals)
    while( pos < shape_str.size() && shape_str[pos] != ',' && shape_str[pos] != ')' )
    {
      ++pos;
    }
  }

  return true;
}

// Decompress zlib-compressed data
std::vector<unsigned char> decompress_zlib( const unsigned char* compressed_data,
                                            size_t compressed_size,
                                            size_t uncompressed_size )
{
  std::vector<unsigned char> result( uncompressed_size );

  z_stream strm;
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.opaque = Z_NULL;
  strm.avail_in = static_cast<uInt>(compressed_size);
  strm.next_in = const_cast<unsigned char*>(compressed_data);
  strm.avail_out = static_cast<uInt>(uncompressed_size);
  strm.next_out = result.data();

  // Use -MAX_WBITS for raw deflate (no zlib header)
  if( inflateInit2(&strm, -MAX_WBITS) != Z_OK )
  {
    return {};
  }

  int ret = inflate(&strm, Z_FINISH);
  inflateEnd(&strm);

  if( ret != Z_STREAM_END )
  {
    return {};
  }

  return result;
}

// Read NPZ file and extract arrays as map of name -> double vector
std::map<std::string, std::vector<double>> read_npz_arrays( const std::string& filename,
                                                             std::map<std::string, std::vector<size_t>>& shapes )
{
  std::map<std::string, std::vector<double>> arrays;

  std::ifstream file( filename, std::ios::binary );
  if( !file )
  {
    return arrays;
  }

  // Read entire file
  file.seekg( 0, std::ios::end );
  size_t file_size = file.tellg();
  file.seekg( 0, std::ios::beg );

  std::vector<unsigned char> file_data( file_size );
  file.read( reinterpret_cast<char*>(file_data.data()), file_size );
  file.close();

  // Parse ZIP entries
  size_t offset = 0;
  while( offset + sizeof(zip_local_file_header) < file_size )
  {
    // Check for local file header signature
    uint32_t sig = read_le<uint32_t>( &file_data[offset] );
    if( sig != 0x04034b50 )
    {
      break;  // No more local file headers
    }

    const zip_local_file_header* hdr =
      reinterpret_cast<const zip_local_file_header*>( &file_data[offset] );

    size_t filename_start = offset + sizeof(zip_local_file_header);
    std::string entry_name( file_data.begin() + filename_start,
                            file_data.begin() + filename_start + hdr->filename_len );

    size_t data_start = filename_start + hdr->filename_len + hdr->extra_len;

    // Get the array data
    std::vector<unsigned char> npy_data;
    if( hdr->compression == 0 )
    {
      // Stored (no compression)
      npy_data.assign( file_data.begin() + data_start,
                       file_data.begin() + data_start + hdr->uncompressed_size );
    }
    else if( hdr->compression == 8 )
    {
      // Deflate compression
      npy_data = decompress_zlib( &file_data[data_start],
                                  hdr->compressed_size,
                                  hdr->uncompressed_size );
    }

    if( !npy_data.empty() )
    {
      // Parse NPY header
      std::vector<size_t> shape;
      bool is_fortran_order;
      char dtype_char;
      size_t dtype_size;
      size_t header_size;

      if( parse_npy_header( npy_data, shape, is_fortran_order, dtype_char, dtype_size, header_size ) )
      {
        // Calculate total number of elements
        size_t num_elements = 1;
        for( size_t dim : shape )
        {
          num_elements *= dim;
        }

        // Extract array data
        std::vector<double> values( num_elements );
        const unsigned char* array_data = npy_data.data() + header_size;

        for( size_t i = 0; i < num_elements; ++i )
        {
          if( dtype_char == 'f' && dtype_size == 8 )
          {
            // float64
            double val;
            std::memcpy( &val, array_data + i * 8, 8 );
            values[i] = val;
          }
          else if( dtype_char == 'f' && dtype_size == 4 )
          {
            // float32
            float val;
            std::memcpy( &val, array_data + i * 4, 4 );
            values[i] = static_cast<double>(val);
          }
        }

        // Remove .npy extension from name if present
        std::string array_name = entry_name;
        if( array_name.size() > 4 && array_name.substr(array_name.size() - 4) == ".npy" )
        {
          array_name = array_name.substr(0, array_name.size() - 4);
        }

        arrays[array_name] = values;
        shapes[array_name] = shape;
      }
    }

    // Move to next entry
    offset = data_start + hdr->compressed_size;
  }

  return arrays;
}
#endif // VIAME_ENABLE_ZLIB

} // end anon namespace

namespace viame {

using namespace kwiver::vital;

static auto logger = get_logger( "viame.camera_rig_io" );

std::string
get_file_ext( path_t const & FN )
{
  std::string ext;
  auto const
    len = FN.length(),
    n = FN.rfind('.', len);
  if (n != std::string::npos)
  {
     ext = FN.substr(n);
  }
  return ext;
}

camera_rig_sptr
read_camera_rig( path_list_t const & cam_files )
{
  camera_rig_sptr rig ( new camera_rig() );
  for ( auto const & cf : cam_files )
  {
    try
    {
      rig->add( cf, read_krtd_file( cf ) );
    }
    catch ( const file_not_found_exception& )
    {
      LOG_ERROR(logger, "error: unable to find " << cf);
      continue;
    }
  }
  if ( rig->empty() )
  {
    VITAL_THROW( invalid_data,
                 "no cameras initialized from the given list of files" ) ;
  }
  return rig;
}

camera_rig_stereo_sptr
read_stereo_rig_json( path_t const& FN )
{
  std::ifstream is(FN);
  cereal::JSONInputArchive ar(is);
  camera_collection cams;
  std::map< std::string, intrinsics_builder > intrinsics_lr;
  std::string LEFT("left"), RIGHT("right");
  auto sides = {LEFT, RIGHT};

  for (const auto& name: sides)
  {
    double fx=1, fy=1;
    ar( cereal::make_nvp( "fx_" + name, fx) );
    ar( cereal::make_nvp( "fy_" + name, fy) );

    double cx=0, cy=0;
    ar ( cereal::make_nvp( "cx_" + name, cx) );
    ar ( cereal::make_nvp( "cy_" + name, cy) );

    // Read distortion coefficients: k1, k2, p1, p2, k3
    Eigen::VectorXd dist(5);
    dist.setZero();
    ar( cereal::make_nvp( "k1_" + name, dist[0] ) );
    ar( cereal::make_nvp( "k2_" + name, dist[1] ) );
    ar( cereal::make_nvp( "p1_" + name, dist[2] ) );
    ar( cereal::make_nvp( "p2_" + name, dist[3] ) );
    // k3 is optional depending on the input
    try {
      ar( cereal::make_nvp( "k3_" + name, dist[4] ) );
    } catch( ... ) {
      dist[4] = 0.0;
    }

    intrinsics_lr[name] = intrinsics_builder( fx, fy, cx, cy, dist );
  }

  vector_3d center = { 0, 0, 0 };
  rotation_d rotation;
  cams[LEFT] = std::make_shared<simple_camera_perspective>(
    center, rotation, intrinsics_lr[LEFT].make_intrinsics()
  );

  std::vector<double> T, R;
  ar( CEREAL_NVP(T) );
  int const n=3;
  vector_3d tv;
  for (int i=0; i<n; ++i)
  {
    tv[i]=T[i];
  }

  ar( CEREAL_NVP(R) );
  Eigen::Matrix<double,3,3> rm;
  unsigned k=0;
  for (int i=0; i<n; ++i)
  {
    for (int j=0; j<n; ++j)
    {
      rm(i,j) = R[k++];
    }
  }
  rotation = rotation_d(rm);
  auto camp = std::make_shared<simple_camera_perspective>(
    center, rotation, intrinsics_lr[RIGHT].make_intrinsics()
  );
  camp->set_translation(tv);
  cams[RIGHT] = camp;

  return std::make_shared<camera_rig_stereo>(
      cams[LEFT], cams[RIGHT]
  );
}

camera_rig_stereo_sptr
read_stereo_rig_yaml( path_t const& FN )
{
  // Parse the YAML file to extract matrices
  auto matrices = parse_ocv_yaml_file( FN );
  if( matrices.empty() )
  {
    LOG_ERROR( logger, "Failed to parse YAML file or no matrices found: " + FN );
    return camera_rig_stereo_sptr();
  }

  // Look for camera matrices (M1/M2 or cameraMatrixL/cameraMatrixR)
  auto find_matrix = [&matrices]( const std::vector<std::string>& names ) -> const ocv_yaml_matrix* {
    for( const auto& name : names )
    {
      auto it = matrices.find( name );
      if( it != matrices.end() && it->second.is_valid() )
      {
        return &it->second;
      }
    }
    return nullptr;
  };

  // Try to find camera matrices
  const auto* M1 = find_matrix( { "M1", "cameraMatrixL", "cameraMatrix1" } );
  const auto* M2 = find_matrix( { "M2", "cameraMatrixR", "cameraMatrix2" } );
  const auto* D1 = find_matrix( { "D1", "distCoeffsL", "distCoeffs1" } );
  const auto* D2 = find_matrix( { "D2", "distCoeffsR", "distCoeffs2" } );
  const auto* R = find_matrix( { "R" } );
  const auto* T = find_matrix( { "T" } );

  // Check that we have the minimum required matrices
  if( !M1 || !M2 )
  {
    LOG_ERROR( logger, "YAML file missing required camera matrices (M1/M2): " + FN );
    return camera_rig_stereo_sptr();
  }

  // Extract left camera intrinsics from 3x3 matrix
  // K = [fx  0  cx]
  //     [0  fy  cy]
  //     [0   0   1]
  if( M1->rows != 3 || M1->cols != 3 || M2->rows != 3 || M2->cols != 3 )
  {
    LOG_ERROR( logger, "Camera matrices must be 3x3: " + FN );
    return camera_rig_stereo_sptr();
  }

  double fx_left = M1->at( 0, 0 );
  double fy_left = M1->at( 1, 1 );
  double cx_left = M1->at( 0, 2 );
  double cy_left = M1->at( 1, 2 );

  double fx_right = M2->at( 0, 0 );
  double fy_right = M2->at( 1, 1 );
  double cx_right = M2->at( 0, 2 );
  double cy_right = M2->at( 1, 2 );

  // Extract distortion coefficients (k1, k2, p1, p2, k3)
  Eigen::VectorXd dist_left( 5 );
  Eigen::VectorXd dist_right( 5 );
  dist_left.setZero();
  dist_right.setZero();

  if( D1 && D1->is_valid() )
  {
    size_t n = std::min( D1->data.size(), size_t( 5 ) );
    for( size_t i = 0; i < n; ++i )
    {
      dist_left[i] = D1->data[i];
    }
  }

  if( D2 && D2->is_valid() )
  {
    size_t n = std::min( D2->data.size(), size_t( 5 ) );
    for( size_t i = 0; i < n; ++i )
    {
      dist_right[i] = D2->data[i];
    }
  }

  // Build intrinsics
  intrinsics_builder left_intrinsics( fx_left, fy_left, cx_left, cy_left, dist_left );
  intrinsics_builder right_intrinsics( fx_right, fy_right, cx_right, cy_right, dist_right );

  // Build left camera (at origin with identity rotation)
  vector_3d center = { 0, 0, 0 };
  rotation_d rotation;
  auto left_cam = std::make_shared<simple_camera_perspective>(
    center, rotation, left_intrinsics.make_intrinsics()
  );

  // Build right camera with rotation and translation relative to left
  Eigen::Matrix<double, 3, 3> rm;
  rm.setIdentity();

  if( R && R->is_valid() && R->rows == 3 && R->cols == 3 )
  {
    for( int i = 0; i < 3; ++i )
    {
      for( int j = 0; j < 3; ++j )
      {
        rm( i, j ) = R->at( i, j );
      }
    }
  }

  vector_3d tv = { 0, 0, 0 };
  if( T && T->is_valid() && T->data.size() >= 3 )
  {
    tv[0] = T->data[0];
    tv[1] = T->data[1];
    tv[2] = T->data[2];
  }

  rotation = rotation_d( rm );
  auto right_cam = std::make_shared<simple_camera_perspective>(
    center, rotation, right_intrinsics.make_intrinsics()
  );
  right_cam->set_translation( tv );

  return std::make_shared<camera_rig_stereo>( left_cam, right_cam );
}

camera_rig_stereo_sptr
read_stereo_rig_from_ocv_dir( path_t const& dir_path )
{
  // OpenCV stereo calibration typically outputs two files:
  // - intrinsics.yml: M1, D1, M2, D2 (camera matrices and distortion coefficients)
  // - extrinsics.yml: R, T, R1, R2, P1, P2, Q (stereo rectification params)

  std::string intrinsics_path = dir_path + "/intrinsics.yml";
  std::string extrinsics_path = dir_path + "/extrinsics.yml";

  // Parse both files
  auto intrinsics = parse_ocv_yaml_file( intrinsics_path );
  auto extrinsics = parse_ocv_yaml_file( extrinsics_path );

  if( intrinsics.empty() && extrinsics.empty() )
  {
    LOG_ERROR( logger, "Failed to read OpenCV calibration files from directory: " + dir_path );
    return camera_rig_stereo_sptr();
  }

  // Merge matrices from both files (extrinsics takes precedence for duplicates)
  std::map<std::string, ocv_yaml_matrix> matrices;
  for( const auto& kv : intrinsics )
  {
    matrices[kv.first] = kv.second;
  }
  for( const auto& kv : extrinsics )
  {
    matrices[kv.first] = kv.second;
  }

  // Find required matrices
  auto find_matrix = [&matrices]( const std::vector<std::string>& names ) -> const ocv_yaml_matrix* {
    for( const auto& name : names )
    {
      auto it = matrices.find( name );
      if( it != matrices.end() && it->second.is_valid() )
      {
        return &it->second;
      }
    }
    return nullptr;
  };

  const auto* M1 = find_matrix( { "M1", "cameraMatrixL", "cameraMatrix1" } );
  const auto* M2 = find_matrix( { "M2", "cameraMatrixR", "cameraMatrix2" } );
  const auto* D1 = find_matrix( { "D1", "distCoeffsL", "distCoeffs1" } );
  const auto* D2 = find_matrix( { "D2", "distCoeffsR", "distCoeffs2" } );
  const auto* R = find_matrix( { "R" } );
  const auto* T = find_matrix( { "T" } );

  if( !M1 || !M2 )
  {
    LOG_ERROR( logger, "OpenCV calibration files missing required camera matrices (M1/M2)" );
    return camera_rig_stereo_sptr();
  }

  if( M1->rows != 3 || M1->cols != 3 || M2->rows != 3 || M2->cols != 3 )
  {
    LOG_ERROR( logger, "Camera matrices must be 3x3" );
    return camera_rig_stereo_sptr();
  }

  // Extract intrinsics
  double fx_left = M1->at( 0, 0 );
  double fy_left = M1->at( 1, 1 );
  double cx_left = M1->at( 0, 2 );
  double cy_left = M1->at( 1, 2 );

  double fx_right = M2->at( 0, 0 );
  double fy_right = M2->at( 1, 1 );
  double cx_right = M2->at( 0, 2 );
  double cy_right = M2->at( 1, 2 );

  // Extract distortion
  Eigen::VectorXd dist_left( 5 );
  Eigen::VectorXd dist_right( 5 );
  dist_left.setZero();
  dist_right.setZero();

  if( D1 && D1->is_valid() )
  {
    size_t n = std::min( D1->data.size(), size_t( 5 ) );
    for( size_t i = 0; i < n; ++i )
    {
      dist_left[i] = D1->data[i];
    }
  }

  if( D2 && D2->is_valid() )
  {
    size_t n = std::min( D2->data.size(), size_t( 5 ) );
    for( size_t i = 0; i < n; ++i )
    {
      dist_right[i] = D2->data[i];
    }
  }

  // Build intrinsics
  intrinsics_builder left_intrinsics( fx_left, fy_left, cx_left, cy_left, dist_left );
  intrinsics_builder right_intrinsics( fx_right, fy_right, cx_right, cy_right, dist_right );

  // Build left camera
  vector_3d center = { 0, 0, 0 };
  rotation_d rotation;
  auto left_cam = std::make_shared<simple_camera_perspective>(
    center, rotation, left_intrinsics.make_intrinsics()
  );

  // Build right camera
  Eigen::Matrix<double, 3, 3> rm;
  rm.setIdentity();

  if( R && R->is_valid() && R->rows == 3 && R->cols == 3 )
  {
    for( int i = 0; i < 3; ++i )
    {
      for( int j = 0; j < 3; ++j )
      {
        rm( i, j ) = R->at( i, j );
      }
    }
  }

  vector_3d tv = { 0, 0, 0 };
  if( T && T->is_valid() && T->data.size() >= 3 )
  {
    tv[0] = T->data[0];
    tv[1] = T->data[1];
    tv[2] = T->data[2];
  }

  rotation = rotation_d( rm );
  auto right_cam = std::make_shared<simple_camera_perspective>(
    center, rotation, right_intrinsics.make_intrinsics()
  );
  right_cam->set_translation( tv );

  return std::make_shared<camera_rig_stereo>( left_cam, right_cam );
}

#ifdef VIAME_ENABLE_ZLIB
camera_rig_stereo_sptr
read_stereo_rig_npz( path_t const& FN )
{
  // Read all arrays from NPZ file
  std::map<std::string, std::vector<size_t>> shapes;
  auto arrays = read_npz_arrays( FN, shapes );

  if( arrays.empty() )
  {
    LOG_ERROR( logger, "Failed to read NPZ file or no arrays found: " + FN );
    return camera_rig_stereo_sptr();
  }

  // Expected arrays in the NPZ file:
  // - R: 3x3 rotation matrix
  // - T: 3x1 translation vector
  // - cameraMatrixL: 3x3 left camera intrinsic matrix
  // - cameraMatrixR: 3x3 right camera intrinsic matrix
  // - distCoeffsL: distortion coefficients for left camera
  // - distCoeffsR: distortion coefficients for right camera

  auto find_array = [&arrays]( const std::string& name ) -> const std::vector<double>* {
    auto it = arrays.find( name );
    return ( it != arrays.end() ) ? &it->second : nullptr;
  };

  const auto* R_arr = find_array( "R" );
  const auto* T_arr = find_array( "T" );
  const auto* K1_arr = find_array( "cameraMatrixL" );
  const auto* K2_arr = find_array( "cameraMatrixR" );
  const auto* dist1_arr = find_array( "distCoeffsL" );
  const auto* dist2_arr = find_array( "distCoeffsR" );

  if( !R_arr || !T_arr || !K1_arr || !K2_arr )
  {
    LOG_ERROR( logger, "NPZ file missing required arrays (R, T, cameraMatrixL, cameraMatrixR)" );
    return camera_rig_stereo_sptr();
  }

  // Extract left camera intrinsics from 3x3 matrix
  // K = [fx  0  cx]
  //     [0  fy  cy]
  //     [0   0   1]
  double fx_left = (*K1_arr)[0];
  double fy_left = (*K1_arr)[4];
  double cx_left = (*K1_arr)[2];
  double cy_left = (*K1_arr)[5];

  // Extract right camera intrinsics
  double fx_right = (*K2_arr)[0];
  double fy_right = (*K2_arr)[4];
  double cx_right = (*K2_arr)[2];
  double cy_right = (*K2_arr)[5];

  // Extract distortion coefficients (k1, k2, p1, p2, k3)
  Eigen::VectorXd dist_left(5);
  Eigen::VectorXd dist_right(5);
  dist_left.setZero();
  dist_right.setZero();

  if( dist1_arr )
  {
    size_t n = std::min( dist1_arr->size(), size_t(5) );
    for( size_t i = 0; i < n; ++i )
    {
      dist_left[i] = (*dist1_arr)[i];
    }
  }

  if( dist2_arr )
  {
    size_t n = std::min( dist2_arr->size(), size_t(5) );
    for( size_t i = 0; i < n; ++i )
    {
      dist_right[i] = (*dist2_arr)[i];
    }
  }

  // Build intrinsics
  intrinsics_builder left_intrinsics( fx_left, fy_left, cx_left, cy_left, dist_left );
  intrinsics_builder right_intrinsics( fx_right, fy_right, cx_right, cy_right, dist_right );

  // Build left camera (at origin with identity rotation)
  vector_3d center = { 0, 0, 0 };
  rotation_d rotation;
  auto left_cam = std::make_shared<simple_camera_perspective>(
    center, rotation, left_intrinsics.make_intrinsics()
  );

  // Extract rotation matrix (3x3, row-major)
  Eigen::Matrix<double, 3, 3> rm;
  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      rm(i, j) = (*R_arr)[i * 3 + j];
    }
  }

  // Extract translation vector
  vector_3d tv;
  tv[0] = (*T_arr)[0];
  tv[1] = (*T_arr)[1];
  tv[2] = (*T_arr)[2];

  // Build right camera with rotation and translation relative to left
  rotation = rotation_d( rm );
  auto right_cam = std::make_shared<simple_camera_perspective>(
    center, rotation, right_intrinsics.make_intrinsics()
  );
  right_cam->set_translation( tv );

  return std::make_shared<camera_rig_stereo>( left_cam, right_cam );
}
#endif // VIAME_ENABLE_ZLIB

// ============================================================================
// MATLAB .mat (Level 5 / v7) reader
//
// Parses numeric variables from a MAT-file into name -> doubles, descending one
// level into a struct (e.g. the Bouguet toolbox "Cal" struct). Handles the
// uncompressed Level-5 layout and, when built with zlib, v7 miCOMPRESSED
// elements. MAT v7.3 (HDF5) is not supported -- matching scipy.io.loadmat's
// default behavior. Implemented by hand (like the .npz reader above) since
// VIAME does not bundle libmatio.
// ============================================================================
namespace {

enum mat_data_type {
  MAT_miINT8 = 1, MAT_miUINT8 = 2, MAT_miINT16 = 3, MAT_miUINT16 = 4,
  MAT_miINT32 = 5, MAT_miUINT32 = 6, MAT_miSINGLE = 7, MAT_miDOUBLE = 9,
  MAT_miMATRIX = 14, MAT_miCOMPRESSED = 15
};
enum mat_array_class { MAT_mxSTRUCT_CLASS = 2, MAT_mxDOUBLE_CLASS = 6 };

struct mat_reader
{
  unsigned char const* end;
  bool swap;

  uint16_t u16( unsigned char const* q ) const
  {
    uint16_t v; std::memcpy( &v, q, 2 );
    return swap ? static_cast< uint16_t >( ( v >> 8 ) | ( v << 8 ) ) : v;
  }
  uint32_t u32( unsigned char const* q ) const
  {
    uint32_t v; std::memcpy( &v, q, 4 );
    if( swap )
    {
      v = ( ( v >> 24 ) & 0xFF ) | ( ( v >> 8 ) & 0xFF00 ) |
          ( ( v << 8 ) & 0xFF0000 ) | ( ( v << 24 ) & 0xFF000000 );
    }
    return v;
  }
  double f64( unsigned char const* q ) const
  {
    uint64_t v; std::memcpy( &v, q, 8 );
    if( swap )
    {
      v = ( ( v & 0xFFULL ) << 56 ) | ( ( v & 0xFF00ULL ) << 40 ) |
          ( ( v & 0xFF0000ULL ) << 24 ) | ( ( v & 0xFF000000ULL ) << 8 ) |
          ( ( v >> 8 ) & 0xFF000000ULL ) | ( ( v >> 24 ) & 0xFF0000ULL ) |
          ( ( v >> 40 ) & 0xFF00ULL ) | ( ( v >> 56 ) & 0xFFULL );
    }
    double d; std::memcpy( &d, &v, 8 ); return d;
  }
};

struct mat_tag
{
  uint32_t type;
  uint32_t nbytes;
  unsigned char const* data;   // payload start
  unsigned char const* next;   // next element (8-byte aligned)
};

bool read_mat_tag( mat_reader const& r, unsigned char const* pos, mat_tag& out )
{
  if( pos + 4 > r.end ) return false;
  uint32_t const w = r.u32( pos );
  uint32_t const hi = w >> 16;
  if( hi != 0 )
  {
    // Small element format: type (low 16) | nbytes (high 16), data in next 4 B
    out.type = w & 0xFFFF;
    out.nbytes = hi;
    out.data = pos + 4;
    out.next = pos + 8;
  }
  else
  {
    if( pos + 8 > r.end ) return false;
    out.type = w;
    out.nbytes = r.u32( pos + 4 );
    out.data = pos + 8;
    out.next = pos + 8 + ( ( out.nbytes + 7u ) & ~7u );  // pad to 8 bytes
  }
  return out.data <= r.end;
}

std::vector< double >
mat_extract_doubles( mat_reader const& r, mat_tag const& t )
{
  std::vector< double > out;
  size_t n = 0;
  switch( t.type )
  {
    case MAT_miDOUBLE:
      n = t.nbytes / 8; out.resize( n );
      for( size_t i = 0; i < n; ++i ) out[ i ] = r.f64( t.data + 8 * i );
      break;
    case MAT_miSINGLE:
      n = t.nbytes / 4; out.resize( n );
      for( size_t i = 0; i < n; ++i )
      { uint32_t b = r.u32( t.data + 4 * i ); float f; std::memcpy( &f, &b, 4 ); out[ i ] = f; }
      break;
    case MAT_miINT32:
      n = t.nbytes / 4; out.resize( n );
      for( size_t i = 0; i < n; ++i ) out[ i ] = static_cast< int32_t >( r.u32( t.data + 4 * i ) );
      break;
    case MAT_miUINT32:
      n = t.nbytes / 4; out.resize( n );
      for( size_t i = 0; i < n; ++i ) out[ i ] = r.u32( t.data + 4 * i );
      break;
    case MAT_miINT16:
      n = t.nbytes / 2; out.resize( n );
      for( size_t i = 0; i < n; ++i ) out[ i ] = static_cast< int16_t >( r.u16( t.data + 2 * i ) );
      break;
    case MAT_miUINT16:
      n = t.nbytes / 2; out.resize( n );
      for( size_t i = 0; i < n; ++i ) out[ i ] = r.u16( t.data + 2 * i );
      break;
    case MAT_miUINT8:
    case MAT_miINT8:
      n = t.nbytes; out.resize( n );
      for( size_t i = 0; i < n; ++i ) out[ i ] = t.data[ i ];
      break;
    default: break;
  }
  return out;
}

// Parse a miMATRIX element into result. For a struct, descends into its fields
// (using the field names as keys). override_name keys a struct field's value.
void
mat_collect( mat_reader const& r, mat_tag const& mt,
             std::map< std::string, std::vector< double > >& result,
             std::string const& override_name )
{
  unsigned char const* p = mt.data;
  unsigned char const* const matend = mt.data + mt.nbytes;

  mat_tag flags; if( !read_mat_tag( r, p, flags ) ) return; p = flags.next;
  uint32_t const cls = r.u32( flags.data ) & 0xFFu;

  mat_tag dims; if( !read_mat_tag( r, p, dims ) ) return; p = dims.next;

  mat_tag nm; if( !read_mat_tag( r, p, nm ) ) return; p = nm.next;
  std::string const name = override_name.empty()
    ? std::string( reinterpret_cast< char const* >( nm.data ), nm.nbytes )
    : override_name;

  if( cls == MAT_mxSTRUCT_CLASS )
  {
    mat_tag fnl; if( !read_mat_tag( r, p, fnl ) ) return; p = fnl.next;
    int const maxlen = static_cast< int32_t >( r.u32( fnl.data ) );
    mat_tag fns; if( !read_mat_tag( r, p, fns ) ) return; p = fns.next;
    int const nfields = maxlen > 0
      ? static_cast< int >( fns.nbytes / static_cast< uint32_t >( maxlen ) ) : 0;
    std::vector< std::string > fnames;
    for( int i = 0; i < nfields; ++i )
    {
      char const* s = reinterpret_cast< char const* >( fns.data ) + static_cast< size_t >( i ) * maxlen;
      size_t len = 0;
      while( len < static_cast< size_t >( maxlen ) && s[ len ] != '\0' ) ++len;
      fnames.emplace_back( s, len );
    }
    for( int i = 0; i < nfields && p + 4 <= matend; ++i )
    {
      mat_tag ft; if( !read_mat_tag( r, p, ft ) ) break;
      if( ft.type == MAT_miMATRIX ) mat_collect( r, ft, result, fnames[ i ] );
      p = ft.next;
    }
  }
  else  // numeric: real part is the next element
  {
    mat_tag pr; if( !read_mat_tag( r, p, pr ) ) return;
    result[ name ] = mat_extract_doubles( r, pr );
  }
}

#ifdef VIAME_ENABLE_ZLIB
// Inflate a standard (zlib-header) stream of unknown output size.
std::vector< unsigned char >
inflate_zlib_stream( unsigned char const* data, size_t n )
{
  std::vector< unsigned char > out;
  z_stream strm; std::memset( &strm, 0, sizeof( strm ) );
  if( inflateInit( &strm ) != Z_OK ) return out;
  strm.next_in = const_cast< Bytef* >( data );
  strm.avail_in = static_cast< uInt >( n );
  std::vector< unsigned char > chunk( 65536 );
  int ret = Z_OK;
  do
  {
    strm.next_out = chunk.data();
    strm.avail_out = static_cast< uInt >( chunk.size() );
    ret = inflate( &strm, Z_NO_FLUSH );
    if( ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR )
    {
      inflateEnd( &strm ); return {};
    }
    out.insert( out.end(), chunk.data(), chunk.data() + ( chunk.size() - strm.avail_out ) );
  }
  while( ret != Z_STREAM_END && strm.avail_in > 0 );
  inflateEnd( &strm );
  return out;
}
#endif

std::map< std::string, std::vector< double > >
read_mat_doubles( path_t const& FN )
{
  std::map< std::string, std::vector< double > > result;
  std::ifstream is( FN, std::ios::binary );
  if( !is ) return result;
  std::vector< unsigned char > buf(
    ( std::istreambuf_iterator< char >( is ) ),
    std::istreambuf_iterator< char >() );
  if( buf.size() < 128 ) return result;

  mat_reader r;
  r.end = buf.data() + buf.size();
  // Endian indicator at bytes 126-127: 'IM' => little-endian (no swap on x86).
  r.swap = ( buf[ 126 ] == 'M' && buf[ 127 ] == 'I' );

  unsigned char const* p = buf.data() + 128;
  while( p + 8 <= r.end )
  {
    mat_tag t; if( !read_mat_tag( r, p, t ) ) break;
    if( t.type == MAT_miMATRIX )
    {
      mat_collect( r, t, result, std::string() );
      p = t.next;
    }
    else if( t.type == MAT_miCOMPRESSED )
    {
#ifdef VIAME_ENABLE_ZLIB
      std::vector< unsigned char > dec = inflate_zlib_stream( t.data, t.nbytes );
      if( dec.size() >= 8 )
      {
        mat_reader r2; r2.end = dec.data() + dec.size(); r2.swap = r.swap;
        mat_tag t2;
        if( read_mat_tag( r2, dec.data(), t2 ) && t2.type == MAT_miMATRIX )
        {
          mat_collect( r2, t2, result, std::string() );
        }
      }
#endif
      // Compressed elements are written without the 8-byte padding that other
      // data elements use, so advance by the exact byte count.
      p = t.data + t.nbytes;
    }
    else
    {
      p = t.next;
    }
  }
  return result;
}

} // anonymous namespace

camera_rig_stereo_sptr
read_stereo_rig_mat( path_t const& FN )
{
  auto vars = read_mat_doubles( FN );
  auto present = [&]( char const* k ) {
    auto it = vars.find( k );
    return it != vars.end() && !it->second.empty();
  };

  // Bouguet stereo calibration fields (flat, or flattened from a "Cal" struct)
  if( !present( "om" ) || !present( "T" ) ||
      !present( "fc_left" ) || !present( "cc_left" ) ||
      !present( "fc_right" ) || !present( "cc_right" ) )
  {
    LOG_ERROR( logger,
      "unable to read stereo rig from .mat (missing Bouguet fields "
      "om/T/fc_*/cc_*): " + FN );
    return camera_rig_stereo_sptr();
  }

  auto build_intrinsics =
    [&]( char const* fc_key, char const* cc_key, char const* kc_key )
    {
      std::vector< double > const& fc = vars[ fc_key ];
      std::vector< double > const& cc = vars[ cc_key ];
      double const fx = fc[ 0 ];
      double const fy = fc.size() > 1 ? fc[ 1 ] : fc[ 0 ];
      double const cx = cc[ 0 ];
      double const cy = cc.size() > 1 ? cc[ 1 ] : 0.0;
      Eigen::VectorXd dist( 5 ); dist.setZero();
      auto it = vars.find( kc_key );
      if( it != vars.end() )
      {
        for( size_t i = 0; i < it->second.size() && i < 5; ++i )
        {
          dist[ static_cast< int >( i ) ] = it->second[ i ];
        }
      }
      return intrinsics_builder( fx, fy, cx, cy, dist );
    };

  intrinsics_builder const il = build_intrinsics( "fc_left", "cc_left", "kc_left" );
  intrinsics_builder const ir = build_intrinsics( "fc_right", "cc_right", "kc_right" );

  vector_3d const center = { 0, 0, 0 };
  camera_collection cams;
  std::string const LEFT( "left" ), RIGHT( "right" );

  cams[ LEFT ] = std::make_shared< simple_camera_perspective >(
    center, rotation_d(), il.make_intrinsics() );

  std::vector< double > const& om = vars[ "om" ];
  std::vector< double > const& T = vars[ "T" ];
  vector_3d const om_vec( om[ 0 ], om[ 1 ], om[ 2 ] );
  rotation_d const rotation( om_vec );  // Rodrigues rotation-vector constructor
  vector_3d const tv( T[ 0 ], T[ 1 ], T[ 2 ] );

  auto camp = std::make_shared< simple_camera_perspective >(
    center, rotation, ir.make_intrinsics() );
  camp->set_translation( tv );
  cams[ RIGHT ] = camp;

  return std::make_shared< camera_rig_stereo >( cams[ LEFT ], cams[ RIGHT ] );
}

camera_rig_stereo_sptr
read_stereo_rig( path_t const& FN )
{
  // Check if the path is a directory (OpenCV calibration format)
  if( kwiversys::SystemTools::FileIsDirectory( FN ) )
  {
    return read_stereo_rig_from_ocv_dir( FN );
  }

  auto const & ext = get_file_ext(FN);
  if (ext == ".json")
  {
    return read_stereo_rig_json(FN);
  }
  else if ( ext == ".yml" || ext == ".yaml")
  {
    return read_stereo_rig_yaml(FN);
  }
  else if ( ext == ".mat" )
  {
    return read_stereo_rig_mat(FN);
  }
#ifdef VIAME_ENABLE_ZLIB
  else if ( ext == ".npz" )
  {
    return read_stereo_rig_npz(FN);
  }
#endif
  else
  {
    LOG_ERROR( logger, "unable to read stereo rig: unsupported extension "+ext );
  }
  return camera_rig_stereo_sptr();
}

void
write_camera_rig( camera_rig_sptr rig )
{
  if (rig == nullptr)
  {
    LOG_ERROR( logger,
     "unable to write: camera rig pointer is null" );
    return;
  }
  for (auto const & c: rig->cameras())
  {
    try
    {
      auto const & cam = dynamic_cast<camera_perspective const&>(*c.second);
      write_krtd_file(cam, c.first);
    }
    catch( std::exception const & e )
    {
      LOG_ERROR(logger, "unable to write " << c.first
          << ": " << e.what() );
    }
  }
}

void
write_stereo_rig_json( camera_rig_stereo_sptr rig, std::string const & FN )
{
  if ( rig == nullptr )
  {
    LOG_ERROR( logger, "unable to write stereo rig: pointer is null" );
    return;
  }
  std::ofstream of( FN );
  cereal::JSONOutputArchive::Options opt(
    32, cereal::JSONOutputArchive::Options::IndentChar::space, 2 );
  cereal::JSONOutputArchive ar( of, opt );
  std::vector< std::string > names = { "left", "right" };
  Eigen::Matrix<double,3,3> Rl;
  Eigen::Matrix<double,3,1> cl;
  for ( auto const & name : names )
  {
    try
    {
      auto const & cam =
        dynamic_cast<camera_perspective const&>( *rig->camera(name) );
      auto const & intr = *cam.intrinsics();
      auto const & f = intr.focal_length();
      auto const & aspect = intr.aspect_ratio();
      auto const & c = intr.principal_point();
      auto const & d = intr.dist_coeffs();
      auto const & dlen = d.size();
      ar( cereal::make_nvp( "fx_" + name, f) );
      ar( cereal::make_nvp( "fy_" + name, f / aspect) );
      ar( cereal::make_nvp( "cx_" + name, c[0]) );
      ar( cereal::make_nvp( "cy_" + name, c[1]) );
      ar( cereal::make_nvp( "k1_" + name, dlen > 0 ? d[0] : 0.0 ) );
      ar( cereal::make_nvp( "k2_" + name, dlen > 1 ? d[1] : 0.0 ) );
      ar( cereal::make_nvp( "p1_" + name, dlen > 2 ? d[2] : 0.0 ) );
      ar( cereal::make_nvp( "p2_" + name, dlen > 3 ? d[3] : 0.0 ) );
      ar( cereal::make_nvp( "k3_" + name, dlen > 4 ? d[4] : 0.0 ) );
      if ( name == "left" )
      {
        Rl = cam.rotation().matrix();
        cl = cam.center();
      }
      else if ( name == "right" )
      {
        // form translation & rotation w.r.t. left
        auto const & Rr = cam.rotation().matrix();
        auto const & rm = Rr * Rl.transpose();
        auto const & tr = cam.translation();
        auto const & tv = tr - Rr * cl;
        auto n = tv.size();
        std::vector<double> T(n);
        for (int i=0; i<n; ++i)
        {
          T[i] = tv[i];
        }
        ar( CEREAL_NVP(T) );
        std::vector<double> R;
        for (int i=0; i<3; ++i)
        {
          for (int j=0; j<3; ++j)
          {
            R.push_back( rm(i,j) );
          }
        }
        ar( CEREAL_NVP(R) );
      }
    }
    catch( std::exception const & e )
    {
      LOG_ERROR(logger, "unable to write " << name
          << ": " << e.what() );
    }
  }
}

void
write_stereo_rig_yaml( camera_rig_stereo_sptr rig, std::string const & FN )
{
  if ( rig == nullptr )
  {
    LOG_ERROR( logger, "unable to write stereo rig: pointer is null" );
    return;
  }
  // TODO write intrinsics and extrinsics using OpenCV FileStorage facility,
  // which likely requires declaring an abstract markup serialization class
  // e.g. in vital/io or vital/algo and its specialization in arrows/ocv
  // that wraps FileStorage functionality for YAML/JSON/XML serialization.
}

void
write_stereo_rig( camera_rig_stereo_sptr rig, std::string const & FN )
{
  auto const & ext = get_file_ext(FN);
  if (ext == ".json")
  {
    write_stereo_rig_json(rig, FN);
  }
  else if ( ext == ".yml" || ext == ".yaml")
  {
    write_stereo_rig_yaml(rig, FN);
  }
}

} // namespace viame
