/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Shared IQR utility classes: descriptor_element, numpy_array_reader,
 *        lsh_index, and distance functions.
 */

#ifndef VIAME_IQR_UTILITIES_IQR_H
#define VIAME_IQR_UTILITIES_IQR_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace viame
{

namespace iqr
{

//--------------------------------------------------------------------------------
// Descriptor element for IQR
struct descriptor_element
{
  std::string uid;
  std::vector< double > vector;

  descriptor_element() = default;
  descriptor_element( const std::string& u, const std::vector< double >& v )
    : uid( u ), vector( v ) {}
};

//--------------------------------------------------------------------------------
// Distance functions
//--------------------------------------------------------------------------------

inline double euclidean_distance( const std::vector< double >& a,
                                  const std::vector< double >& b )
{
  if( a.size() != b.size() )
  {
    return std::numeric_limits< double >::max();
  }

  double sum = 0.0;
  for( size_t i = 0; i < a.size(); ++i )
  {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt( sum );
}

// Cosine distance
// Returns value between 0.0 (identical) and 1.0 (orthogonal) for positive vectors
inline double cosine_distance( const std::vector< double >& a,
                               const std::vector< double >& b )
{
  if( a.size() != b.size() || a.empty() )
  {
    return 1.0;
  }

  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
  for( size_t i = 0; i < a.size(); ++i )
  {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  if( norm_a == 0.0 || norm_b == 0.0 )
  {
    return 1.0;
  }

  double similarity = dot / ( std::sqrt( norm_a ) * std::sqrt( norm_b ) );
  // Clamp to [-1, 1] for numerical stability
  similarity = std::max( -1.0, std::min( 1.0, similarity ) );
  // For positive vectors, convert similarity to distance in [0, 1]
  return std::acos( similarity ) / M_PI;
}

// Histogram intersection distance
// Returns value between 0.0 (full intersection) and 1.0 (no intersection)
// Formula: 1.0 - sum(min(a[i], b[i])) = 1.0 - ((a + b - |a - b|) / 2).sum()
inline double histogram_intersection_distance( const std::vector< double >& a,
                                                const std::vector< double >& b )
{
  if( a.size() != b.size() )
  {
    return 1.0;  // No intersection if sizes differ
  }

  double intersection_sum = 0.0;
  for( size_t i = 0; i < a.size(); ++i )
  {
    // Non-branching min: (a + b - |a - b|) / 2
    intersection_sum += ( a[i] + b[i] - std::abs( a[i] - b[i] ) ) * 0.5;
  }

  return 1.0 - intersection_sum;
}

//--------------------------------------------------------------------------------
// Simple numpy file reader for 1D and 2D arrays
// Supports float64, complex128, and uint8 types (sufficient for ITQ model and hash codes)
class numpy_array_reader
{
public:
  // Parse dtype descriptor from numpy header
  // Returns: "f8" for float64, "c16" for complex128, "u1" for uint8, etc.
  static std::string parse_dtype( const std::string& header )
  {
    // Look for 'descr': '<f8' or 'descr': '<c16' etc.
    size_t descr_start = header.find( "'descr':" );
    if( descr_start == std::string::npos )
    {
      descr_start = header.find( "\"descr\":" );
    }
    if( descr_start == std::string::npos )
    {
      return "";
    }

    // Find the quote after the colon
    size_t quote_start = header.find_first_of( "'\"", descr_start + 8 );
    if( quote_start == std::string::npos )
    {
      return "";
    }

    char quote_char = header[quote_start];
    size_t quote_end = header.find( quote_char, quote_start + 1 );
    if( quote_end == std::string::npos )
    {
      return "";
    }

    std::string descr = header.substr( quote_start + 1, quote_end - quote_start - 1 );

    // Strip byte order prefix (<, >, |, =)
    if( !descr.empty() && ( descr[0] == '<' || descr[0] == '>' ||
                            descr[0] == '|' || descr[0] == '=' ) )
    {
      descr = descr.substr( 1 );
    }

    return descr;
  }

  static bool read_float64_array( const std::string& filepath,
                                   std::vector< double >& out_data,
                                   std::vector< size_t >& out_shape )
  {
    std::ifstream file( filepath, std::ios::binary );
    if( !file.is_open() )
    {
      return false;
    }

    // Read numpy header
    char magic[6];
    file.read( magic, 6 );
    if( magic[0] != '\x93' || std::string( magic + 1, 5 ) != "NUMPY" )
    {
      return false;
    }

    uint8_t major_version, minor_version;
    file.read( reinterpret_cast< char* >( &major_version ), 1 );
    file.read( reinterpret_cast< char* >( &minor_version ), 1 );

    uint32_t header_len;
    if( major_version == 1 )
    {
      uint16_t len16;
      file.read( reinterpret_cast< char* >( &len16 ), 2 );
      header_len = len16;
    }
    else
    {
      file.read( reinterpret_cast< char* >( &header_len ), 4 );
    }

    std::string header( header_len, '\0' );
    file.read( &header[0], header_len );

    // Parse dtype: float64 (f8) and complex128 (c16, real part kept) are
    // what the ITQ model files use; float32 (f4) is what per-video
    // descriptor bundles store to halve their size.
    std::string dtype = parse_dtype( header );
    bool is_complex128 = ( dtype == "c16" );
    bool is_float32 = ( dtype == "f4" );
    if( !is_complex128 && !is_float32 && dtype != "f8" )
    {
      return false;
    }

    // Parse shape from header (simple parsing for common cases)
    out_shape.clear();
    size_t shape_start = header.find( "\'shape\': (" );
    if( shape_start != std::string::npos )
    {
      shape_start += 10;
      size_t shape_end = header.find( ")", shape_start );
      std::string shape_str = header.substr( shape_start, shape_end - shape_start );

      std::stringstream ss( shape_str );
      std::string token;
      while( std::getline( ss, token, ',' ) )
      {
        token.erase( 0, token.find_first_not_of( " " ) );
        token.erase( token.find_last_not_of( " " ) + 1 );
        if( !token.empty() )
        {
          out_shape.push_back( std::stoull( token ) );
        }
      }
    }

    // Calculate total elements
    size_t total_elements = 1;
    for( size_t dim : out_shape )
    {
      total_elements *= dim;
    }

    // Read data based on dtype
    if( is_complex128 )
    {
      // complex128: 16 bytes per element (8 real + 8 imag)
      // Read all data, then extract only real parts
      std::vector< double > raw_data( total_elements * 2 );
      file.read( reinterpret_cast< char* >( raw_data.data() ),
                 total_elements * 2 * sizeof( double ) );

      out_data.resize( total_elements );
      for( size_t i = 0; i < total_elements; ++i )
      {
        out_data[i] = raw_data[i * 2];  // Take only real part, skip imaginary
      }
    }
    else if( is_float32 )
    {
      // float32: 4 bytes per element, widened to double
      std::vector< float > raw_data( total_elements );
      file.read( reinterpret_cast< char* >( raw_data.data() ),
                 total_elements * sizeof( float ) );
      out_data.assign( raw_data.begin(), raw_data.end() );
    }
    else
    {
      // float64: 8 bytes per element
      out_data.resize( total_elements );
      file.read( reinterpret_cast< char* >( out_data.data() ),
                 total_elements * sizeof( double ) );
    }

    return file.good() || file.eof();
  }

  /// Read the uid list that accompanies a descriptor or hash array: one uid
  /// per line, in row order.
  static bool read_uid_list( const std::string& filepath,
                             std::vector< std::string >& out_uids )
  {
    std::ifstream uid_file( filepath );
    if( !uid_file.is_open() )
    {
      return false;
    }
    std::string line;
    while( std::getline( uid_file, line ) )
    {
      line.erase( 0, line.find_first_not_of( " \t\r\n" ) );
      line.erase( line.find_last_not_of( " \t\r\n" ) + 1 );
      if( !line.empty() )
      {
        out_uids.push_back( line );
      }
    }
    return true;
  }

  /// True when `filename` ends with `postfix`.
  static bool ends_with( const std::string& filename, const std::string& postfix )
  {
    return filename.size() >= postfix.size() &&
      filename.compare( filename.size() - postfix.size(),
                        postfix.size(), postfix ) == 0;
  }

  static bool read_uint8_array( const std::string& filepath,
                                 std::vector< uint8_t >& out_data,
                                 std::vector< size_t >& out_shape )
  {
    std::ifstream file( filepath, std::ios::binary );
    if( !file.is_open() )
    {
      return false;
    }

    // Read numpy header
    char magic[6];
    file.read( magic, 6 );
    if( magic[0] != '\x93' || std::string( magic + 1, 5 ) != "NUMPY" )
    {
      return false;
    }

    uint8_t major_version, minor_version;
    file.read( reinterpret_cast< char* >( &major_version ), 1 );
    file.read( reinterpret_cast< char* >( &minor_version ), 1 );

    uint32_t header_len;
    if( major_version == 1 )
    {
      uint16_t len16;
      file.read( reinterpret_cast< char* >( &len16 ), 2 );
      header_len = len16;
    }
    else
    {
      file.read( reinterpret_cast< char* >( &header_len ), 4 );
    }

    std::string header( header_len, '\0' );
    file.read( &header[0], header_len );

    // Parse shape from header
    out_shape.clear();
    size_t shape_start = header.find( "\'shape\': (" );
    if( shape_start != std::string::npos )
    {
      shape_start += 10;
      size_t shape_end = header.find( ")", shape_start );
      std::string shape_str = header.substr( shape_start, shape_end - shape_start );

      std::stringstream ss( shape_str );
      std::string token;
      while( std::getline( ss, token, ',' ) )
      {
        token.erase( 0, token.find_first_not_of( " " ) );
        token.erase( token.find_last_not_of( " " ) + 1 );
        if( !token.empty() )
        {
          out_shape.push_back( std::stoull( token ) );
        }
      }
    }

    // Calculate total elements
    size_t total_elements = 1;
    for( size_t dim : out_shape )
    {
      total_elements *= dim;
    }

    // Read data
    out_data.resize( total_elements );
    file.read( reinterpret_cast< char* >( out_data.data() ), total_elements );

    return file.good() || file.eof();
  }
};

//--------------------------------------------------------------------------------
// LSH Index class for fast approximate nearest neighbor search
// Uses ITQ (Iterative Quantization) for hash code computation
class lsh_index
{
public:
  lsh_index() : m_bit_length( 0 ), m_num_descriptors( 0 ) {}

  bool load( const std::string& hash_codes_file,
             const std::string& hash_uids_file,
             const std::string& mean_vec_file,
             const std::string& rotation_file,
             unsigned bit_length )
  {
    if( !load_model( mean_vec_file, rotation_file, bit_length ) )
    {
      return false;
    }
    if( !append_hashes( hash_codes_file, hash_uids_file ) )
    {
      return false;
    }
    finalize();
    return is_loaded();
  }

  /// Load the ITQ model and every per-video hash file
  /// (`<name><hashes_postfix>` + `<name><uids_postfix>`) found in `dir`.
  /// Returns false when the model is unreadable or no hash file was found.
  bool load_from_dir( const std::string& dir,
                      const std::string& hashes_postfix,
                      const std::string& uids_postfix,
                      const std::string& mean_vec_file,
                      const std::string& rotation_file,
                      unsigned bit_length )
  {
    if( !load_model( mean_vec_file, rotation_file, bit_length ) )
    {
      return false;
    }
    std::vector< std::string > hash_files;
    if( std::filesystem::is_directory( dir ) )
    {
      for( auto const& entry : std::filesystem::directory_iterator( dir ) )
      {
        if( entry.is_regular_file() &&
            numpy_array_reader::ends_with( entry.path().filename().string(),
                                           hashes_postfix ) )
        {
          hash_files.push_back( entry.path().string() );
        }
      }
    }
    // Deterministic order so uid indices are stable between runs
    std::sort( hash_files.begin(), hash_files.end() );
    for( auto const& hash_file : hash_files )
    {
      std::string uid_file =
        hash_file.substr( 0, hash_file.size() - hashes_postfix.size() ) + uids_postfix;
      if( !append_hashes( hash_file, uid_file ) )
      {
        return false;
      }
    }
    finalize();
    return !hash_files.empty() && is_loaded();
  }

  /// Load the ITQ mean vector and rotation matrix, resetting any hash codes.
  bool load_model( const std::string& mean_vec_file,
                   const std::string& rotation_file,
                   unsigned bit_length )
  {
    m_bit_length = bit_length;
    m_hash_codes.clear();
    m_uids.clear();
    m_num_descriptors = 0;
    // Load ITQ mean vector
    std::vector< size_t > mean_shape;
    if( !numpy_array_reader::read_float64_array( mean_vec_file, m_mean_vec, mean_shape ) )
    {
      return false;
    }
    // Load ITQ rotation matrix
    std::vector< size_t > rotation_shape;
    if( !numpy_array_reader::read_float64_array( rotation_file, m_rotation, rotation_shape ) )
    {
      return false;
    }
    if( rotation_shape.size() != 2 )
    {
      return false;
    }
    m_feature_dim = rotation_shape[0];
    m_rotation_cols = rotation_shape[1];
    return true;
  }

  /// Append one hash code array (N x bit_length uint8) and its uid list.
  bool append_hashes( const std::string& hash_codes_file,
                      const std::string& hash_uids_file )
  {
    std::vector< uint8_t > codes;
    std::vector< size_t > hash_shape;
    if( !numpy_array_reader::read_uint8_array( hash_codes_file, codes, hash_shape ) )
    {
      return false;
    }
    if( hash_shape.size() != 2 || hash_shape[1] != m_bit_length )
    {
      return false;
    }
    std::vector< std::string > uids;
    if( !numpy_array_reader::read_uid_list( hash_uids_file, uids ) )
    {
      return false;
    }
    if( uids.size() != hash_shape[0] )
    {
      return false;
    }
    m_hash_codes.insert( m_hash_codes.end(), codes.begin(), codes.end() );
    m_uids.insert( m_uids.end(), uids.begin(), uids.end() );
    m_num_descriptors += hash_shape[0];
    return true;
  }

  /// Rebuild the lookup maps once every hash code has been appended.
  void finalize()
  {
    // Build UID to index map for fast lookups
    m_uid_to_index.clear();
    for( size_t i = 0; i < m_uids.size(); ++i )
    {
      m_uid_to_index[m_uids[i]] = i;
    }
    // Build hash-to-UIDs mapping (groups UIDs by their hash code)
    // This groups all UIDs that have the same hash code
    m_hash_to_uids.clear();
    for( size_t i = 0; i < m_num_descriptors; ++i )
    {
      // Get hash code for this descriptor
      std::vector< uint8_t > hash( m_bit_length );
      for( size_t j = 0; j < m_bit_length; ++j )
      {
        hash[j] = m_hash_codes[i * m_bit_length + j];
      }
      std::string hash_key = hash_to_string( hash );
      m_hash_to_uids[hash_key].push_back( m_uids[i] );
    }
  }

  bool is_loaded() const
  {
    return m_num_descriptors > 0 && !m_hash_codes.empty();
  }

  // Compute hash code for a descriptor using ITQ
  std::vector< uint8_t > compute_hash( const std::vector< double >& descriptor ) const
  {
    if( descriptor.size() != m_feature_dim )
    {
      return {};
    }

    // Center the descriptor
    std::vector< double > centered( m_feature_dim );
    for( size_t i = 0; i < m_feature_dim; ++i )
    {
      centered[i] = descriptor[i] - m_mean_vec[i];
    }

    // Project using rotation matrix: z = centered @ rotation
    std::vector< double > projected( m_rotation_cols, 0.0 );
    for( size_t j = 0; j < m_rotation_cols; ++j )
    {
      for( size_t i = 0; i < m_feature_dim; ++i )
      {
        projected[j] += centered[i] * m_rotation[i * m_rotation_cols + j];
      }
    }

    // Quantize to binary
    std::vector< uint8_t > hash( m_bit_length );
    for( size_t i = 0; i < m_bit_length && i < projected.size(); ++i )
    {
      hash[i] = ( projected[i] >= 0 ) ? 1 : 0;
    }

    return hash;
  }

  // Compute hamming distance between two hash codes
  static unsigned hamming_distance( const std::vector< uint8_t >& a,
                                     const std::vector< uint8_t >& b )
  {
    unsigned dist = 0;
    size_t len = std::min( a.size(), b.size() );
    for( size_t i = 0; i < len; ++i )
    {
      if( a[i] != b[i] )
      {
        ++dist;
      }
    }
    return dist;
  }

  // Find k nearest neighbors using LSH (hamming distance)
  // Returns (uid, hamming_distance) pairs sorted by distance
  std::vector< std::pair< std::string, unsigned > > find_neighbors_lsh(
    const std::vector< double >& query_descriptor,
    size_t k ) const
  {
    if( !is_loaded() )
    {
      return {};
    }

    // Compute query hash
    std::vector< uint8_t > query_hash = compute_hash( query_descriptor );
    if( query_hash.empty() )
    {
      return {};
    }

    // Find k nearest by hamming distance using a max-heap
    using pair_type = std::pair< unsigned, size_t >; // (distance, index)
    std::priority_queue< pair_type > pq;

    for( size_t i = 0; i < m_num_descriptors; ++i )
    {
      // Get hash code for this descriptor
      std::vector< uint8_t > hash( m_bit_length );
      for( size_t j = 0; j < m_bit_length; ++j )
      {
        hash[j] = m_hash_codes[i * m_bit_length + j];
      }

      unsigned dist = hamming_distance( query_hash, hash );

      if( pq.size() < k )
      {
        pq.push( { dist, i } );
      }
      else if( dist < pq.top().first )
      {
        pq.pop();
        pq.push( { dist, i } );
      }
    }

    // Convert to result vector
    std::vector< std::pair< std::string, unsigned > > results;
    results.reserve( pq.size() );
    while( !pq.empty() )
    {
      results.emplace_back( m_uids[pq.top().second], pq.top().first );
      pq.pop();
    }

    // Reverse to get closest first
    std::reverse( results.begin(), results.end() );
    return results;
  }

  // Get UID by index
  const std::string& get_uid( size_t index ) const
  {
    return m_uids[index];
  }

  // Check if UID exists in index
  bool has_uid( const std::string& uid ) const
  {
    return m_uid_to_index.find( uid ) != m_uid_to_index.end();
  }

  size_t size() const { return m_num_descriptors; }
  unsigned bit_length() const { return m_bit_length; }

  // Convert hash code to string key for hash table (supports full 256 bits)
  static std::string hash_to_string( const std::vector< uint8_t >& hash )
  {
    std::string result;
    result.reserve( hash.size() );
    for( uint8_t bit : hash )
    {
      result += ( bit ? '1' : '0' );
    }
    return result;
  }

  // LSH NN search: get n unique hashes, expand to all UIDs
  // Groups results by hash code before expanding to individual descriptors
  std::vector< std::string > find_neighbors_by_hash(
    const std::vector< double >& query_descriptor,
    size_t n ) const
  {
    if( !is_loaded() )
    {
      return {};
    }

    // Compute query hash
    std::vector< uint8_t > query_hash = compute_hash( query_descriptor );
    if( query_hash.empty() )
    {
      return {};
    }

    // Get all unique hashes with their hamming distances to query
    // pair: (hamming_distance, hash_string_key)
    std::vector< std::pair< unsigned, std::string > > hash_distances;
    hash_distances.reserve( m_hash_to_uids.size() );

    for( const auto& entry : m_hash_to_uids )
    {
      // Reconstruct hash vector from string key
      std::vector< uint8_t > hash( entry.first.size() );
      for( size_t i = 0; i < entry.first.size(); ++i )
      {
        hash[i] = ( entry.first[i] == '1' ) ? 1 : 0;
      }

      unsigned dist = hamming_distance( query_hash, hash );
      hash_distances.emplace_back( dist, entry.first );
    }

    // Sort by hamming distance
    std::sort( hash_distances.begin(), hash_distances.end() );

    // Take n nearest unique hashes
    size_t num_hashes = std::min( n, hash_distances.size() );

    // Expand to all UIDs
    std::vector< std::string > neighbor_uids;
    for( size_t i = 0; i < num_hashes; ++i )
    {
      const std::string& hash_key = hash_distances[i].second;
      auto it = m_hash_to_uids.find( hash_key );
      if( it != m_hash_to_uids.end() )
      {
        for( const std::string& uid : it->second )
        {
          neighbor_uids.push_back( uid );
        }
      }
    }

    return neighbor_uids;
  }

private:
  unsigned m_bit_length;
  size_t m_num_descriptors;
  size_t m_feature_dim;
  size_t m_rotation_cols;

  std::vector< double > m_mean_vec;
  std::vector< double > m_rotation;
  std::vector< uint8_t > m_hash_codes;
  std::vector< std::string > m_uids;
  std::unordered_map< std::string, size_t > m_uid_to_index;

  // Hash-to-UIDs mapping (groups UIDs by hash code)
  // Key is string representation of hash (e.g., "0110...1")
  std::unordered_map< std::string, std::vector< std::string > > m_hash_to_uids;
};

} // end namespace iqr
} // end namespace viame

#endif // VIAME_IQR_UTILITIES_IQR_H
