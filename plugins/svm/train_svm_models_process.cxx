/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Train SVM models from descriptor index and label files
 */

#include "train_svm_models_process.h"

#include <vital/vital_types.h>
#include <vital/logger/logger.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <svm.h>

#ifdef VIAME_ENABLE_CPPDB
#include <cppdb/frontend.h>
#endif

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32) || defined(_WIN64)
#include <direct.h>
#define MKDIR(x) _mkdir(x)
#define POPEN(cmd, mode) _popen(cmd, mode)
#define PCLOSE(fp) _pclose(fp)
#else
#include <sys/stat.h>
#define MKDIR(x) mkdir(x, 0755)
#define POPEN(cmd, mode) popen(cmd, mode)
#define PCLOSE(fp) pclose(fp)
#endif

namespace viame
{

namespace svm
{

// Config traits
create_config_trait( descriptor_index_file, std::string, "",
  "Path to CSV file containing descriptor index (uid,v1,v2,...). "
  "Used if conn_str is empty." );
create_config_trait( conn_str, std::string, "",
  "Database connection string (e.g., postgresql:host=localhost;user=postgres). "
  "If set, descriptors are loaded from the DESCRIPTOR table instead of CSV file." );
create_config_trait( label_folder, std::string, "database",
  "Folder containing label files with descriptor UIDs per category" );
create_config_trait( label_extension, std::string, "lbl",
  "File extension for label files" );
create_config_trait( output_directory, std::string, "category_models",
  "Output directory for trained SVM model files" );
create_config_trait( background_category, std::string, "background",
  "Category name to skip (background class)" );
create_config_trait( maximum_positive_count, unsigned, "75",
  "Maximum number of positive samples per category" );
create_config_trait( maximum_negative_count, unsigned, "750",
  "Maximum number of negative samples per category" );
create_config_trait( minimum_positive_threshold, unsigned, "1",
  "Minimum number of positive samples required to train a category model" );
create_config_trait( minimum_negative_threshold, unsigned, "1",
  "Minimum number of negative samples required to train a category model" );
create_config_trait( train_on_neighbors_only, bool, "false",
  "If true, only use nearest-neighbor hard negatives. "
  "If false, use 50/50 split of NN hard negatives and random negatives." );
create_config_trait( two_stage_training, bool, "true",
  "If true, train initial SVM to find hard negatives, then retrain. "
  "If false, use single-stage training with NN-based hard negatives." );
create_config_trait( svm_rerank_negatives, bool, "true",
  "If true (default), train an intermediate SVM to re-rank and select hard negatives. "
  "If false, use only centroid similarity for hard negative selection. "
  "Only applies when two_stage_training is true." );
create_config_trait( auto_compute_neighbors, bool, "true",
  "If true, automatically compute pos_seed_neighbors as "
  "maximum_negative_count / maximum_positive_count. "
  "If false, use the manually specified pos_seed_neighbors value." );
create_config_trait( pos_seed_neighbors, unsigned, "10",
  "Number of nearest neighbors per positive sample for hard negative mining. "
  "Only used if auto_compute_neighbors is false." );
create_config_trait( min_pos_seed_neighbors, unsigned, "2",
  "Minimum number of neighbors per positive sample. "
  "Enforced even when auto_compute_neighbors is true." );
create_config_trait( feedback_sample_count, unsigned, "0",
  "Number of uncertain samples (near decision boundary) to include as additional "
  "hard negatives. Set to 0 to disable." );
create_config_trait( svm_kernel_type, std::string, "linear",
  "SVM kernel type: linear, poly, rbf, sigmoid." );
create_config_trait( svm_c, double, "1.0",
  "SVM regularization parameter C" );
create_config_trait( svm_gamma, double, "0.001",
  "SVM gamma parameter for rbf/poly/sigmoid kernels" );
create_config_trait( normalize_descriptors, bool, "true",
  "If true (default), L2-normalize descriptor vectors before training and scoring. "
  "This is recommended for most descriptor types." );
create_config_trait( score_normalization, std::string, "sigmoid",
  "How to convert SVM output to [0,1] scores: "
  "'sigmoid' (default) uses 1/(1+exp(-decision_value)), "
  "'probability' uses libsvm's probability estimates (requires more training time)." );
create_config_trait( use_class_weights, bool, "false",
  "If true, weight SVM classes inversely proportional to their frequency "
  "to handle class imbalance" );
create_config_trait( distance_metric, std::string, "euclidean",
  "Distance metric for nearest neighbor search: euclidean or cosine" );
create_config_trait( random_seed, int, "0",
  "Random seed for reproducibility. Use -1 for random initialization." );
create_config_trait( nn_index_type, std::string, "brute_force",
  "Nearest neighbor index type: 'brute_force' (default) or 'lsh'. "
  "LSH uses ITQ locality-sensitive hashing for faster approximate NN search." );
create_config_trait( lsh_model_dir, std::string, "",
  "Directory containing ITQ model files from generate_nn_index.py."
  "Required when nn_index_type is 'lsh'." );
create_config_trait( lsh_bit_length, unsigned, "256",
  "Number of bits in the ITQ hash code. Must match the trained model." );
create_config_trait( lsh_itq_iterations, unsigned, "100",
  "Number of ITQ iterations used when training the model." );
create_config_trait( lsh_random_seed, int, "0",
  "Random seed used when training the ITQ model." );
create_config_trait( lsh_hash_ratio, double, "0.2",
  "Fraction of hash codes to search when using LSH. "
  "Higher values give better recall but slower search. "
  "Default is 0.2 (search top 20% of hash codes by Hamming distance)." );

//--------------------------------------------------------------------------------
// Private implementation class
class train_svm_models_process::priv
{
public:
  priv()
    : m_descriptor_index_file( "" )
    , m_conn_str( "" )
    , m_label_folder( "database" )
    , m_label_extension( "lbl" )
    , m_output_directory( "category_models" )
    , m_background_category( "background" )
    , m_maximum_positive_count( 75 )
    , m_maximum_negative_count( 750 )
    , m_minimum_positive_threshold( 1 )
    , m_minimum_negative_threshold( 1 )
    , m_train_on_neighbors_only( false )
    , m_two_stage_training( true )
    , m_svm_rerank_negatives( true )
    , m_auto_compute_neighbors( true )
    , m_pos_seed_neighbors( 10 )
    , m_min_pos_seed_neighbors( 2 )
    , m_feedback_sample_count( 0 )
    , m_svm_kernel_type( "linear" )
    , m_svm_c( 1.0 )
    , m_svm_gamma( 0.001 )
    , m_normalize_descriptors( true )
    , m_score_normalization( "sigmoid" )
    , m_use_class_weights( false )
    , m_distance_metric( "euclidean" )
    , m_random_seed( 0 )
    , m_nn_index_type( "brute_force" )
    , m_lsh_model_dir( "" )
    , m_lsh_bit_length( 256 )
    , m_lsh_itq_iterations( 100 )
    , m_lsh_random_seed( 0 )
    , m_lsh_hash_ratio( 0.2 )
    , m_training_complete( false )
    , m_lsh_initialized( false )
  {}

  ~priv() {}

  std::string m_descriptor_index_file;
  std::string m_conn_str;
  std::string m_label_folder;
  std::string m_label_extension;
  std::string m_output_directory;
  std::string m_background_category;
  unsigned m_maximum_positive_count;
  unsigned m_maximum_negative_count;
  unsigned m_minimum_positive_threshold;
  unsigned m_minimum_negative_threshold;
  bool m_train_on_neighbors_only;
  bool m_two_stage_training;
  bool m_svm_rerank_negatives;
  bool m_auto_compute_neighbors;
  unsigned m_pos_seed_neighbors;
  unsigned m_min_pos_seed_neighbors;
  unsigned m_feedback_sample_count;
  std::string m_svm_kernel_type;
  double m_svm_c;
  double m_svm_gamma;
  bool m_normalize_descriptors;
  std::string m_score_normalization;
  bool m_use_class_weights;
  std::string m_distance_metric;
  int m_random_seed;
  std::string m_nn_index_type;
  std::string m_lsh_model_dir;
  unsigned m_lsh_bit_length;
  unsigned m_lsh_itq_iterations;
  int m_lsh_random_seed;
  double m_lsh_hash_ratio;
  bool m_training_complete;
  bool m_lsh_initialized;

  // Descriptor index
  std::unordered_map< std::string, std::vector< double > > m_descriptor_index;

  // LSH index data
  std::vector< double > m_lsh_mean_vec;
  std::vector< std::vector< double > > m_lsh_rotation;
  std::unordered_map< std::string, std::vector< bool > > m_lsh_hash_codes;
  std::unordered_map< std::vector< bool >, std::vector< std::string > > m_hash_to_uids;

  kwiver::vital::logger_handle_t m_logger;

  // Helper functions
  void load_descriptor_index();
  void load_descriptor_index_from_csv();
#ifdef VIAME_ENABLE_CPPDB
  void load_descriptor_index_from_db();
#endif
  std::unordered_set< std::string > load_uid_file( const std::string& filepath );
  void train_all_models();
  void train_svm_model(
    const std::string& category,
    const std::unordered_set< std::string >& positive_uids,
    const std::unordered_set< std::string >& negative_uids,
    const std::string& output_file );
  int get_kernel_type() const;

  // SVM training helper
  svm_model* train_svm(
    const std::vector< std::pair< std::string, std::vector< double > > >& pos_descriptors,
    const std::vector< std::pair< std::string, std::vector< double > > >& neg_descriptors );

  // Score descriptors using SVM
  std::vector< std::pair< std::string, double > > score_with_svm(
    svm_model* model,
    const std::unordered_set< std::string >& uids_to_score,
    const std::unordered_set< std::string >& exclude_uids ) const;

  // Score by centroid similarity
  std::vector< std::pair< std::string, double > > score_by_centroid_similarity(
    const std::vector< std::pair< std::string, std::vector< double > > >& pos_descriptors,
    const std::unordered_set< std::string >& uids_to_score,
    const std::unordered_set< std::string >& exclude_uids ) const;

  std::vector< std::pair< std::string, double > > find_nearest_neighbors(
    const std::vector< double >& query,
    size_t k ) const;

  std::vector< std::pair< std::string, double > > find_nearest_neighbors_multi(
    const std::vector< std::pair< std::string, std::vector< double > > >& queries,
    size_t k_per_query ) const;

  static double euclidean_distance(
    const std::vector< double >& a,
    const std::vector< double >& b );

  static double cosine_distance(
    const std::vector< double >& a,
    const std::vector< double >& b );

  double compute_distance(
    const std::vector< double >& a,
    const std::vector< double >& b ) const;

  std::mt19937 get_random_generator() const;

  static std::vector< double > normalize_vector( const std::vector< double >& v );

  static bool directory_exists( const std::string& path );
  static bool create_directory( const std::string& path );
  static std::vector< std::string > list_files(
    const std::string& folder,
    const std::string& extension );

  // LSH/ITQ functions
  bool load_itq_model();
  void initialize_lsh_index();
  std::vector< bool > compute_hash( const std::vector< double >& descriptor ) const;
  static unsigned hamming_distance( const std::vector< bool >& a, const std::vector< bool >& b );
  std::vector< std::pair< std::string, double > > find_nearest_neighbors_lsh(
    const std::vector< double >& query,
    size_t k ) const;
  std::vector< std::pair< std::string, double > > find_nearest_neighbors_lsh_multi(
    const std::vector< std::pair< std::string, std::vector< double > > >& queries,
    size_t k_per_query ) const;
  bool load_npy_vector( const std::string& filepath, std::vector< double >& out );
  bool load_npy_matrix( const std::string& filepath, std::vector< std::vector< double > >& out );
  bool load_precomputed_hashes();
  bool load_npy_hash_matrix( const std::string& filepath,
    std::vector< std::vector< bool > >& out, size_t& n_rows, size_t& n_cols );
};


// -----------------------------------------------------------------------------
void
train_svm_models_process::priv
::load_descriptor_index()
{
#ifdef VIAME_ENABLE_CPPDB
  if( !m_conn_str.empty() )
  {
    load_descriptor_index_from_db();
    return;
  }
#endif

  load_descriptor_index_from_csv();
}


// -----------------------------------------------------------------------------
void
train_svm_models_process::priv
::load_descriptor_index_from_csv()
{
  if( m_descriptor_index_file.empty() )
  {
    LOG_ERROR( m_logger, "Either descriptor_index_file or conn_str must be provided" );
    return;
  }

  std::ifstream file( m_descriptor_index_file );
  if( !file.is_open() )
  {
    LOG_ERROR( m_logger, "Failed to open descriptor index file: " << m_descriptor_index_file );
    return;
  }

  std::string line;
  while( std::getline( file, line ) )
  {
    if( line.empty() ) continue;

    std::istringstream ss( line );
    std::string uid;

    if( !std::getline( ss, uid, ',' ) ) continue;

    std::vector< double > values;
    std::string value_str;
    while( std::getline( ss, value_str, ',' ) )
    {
      try
      {
        values.push_back( std::stod( value_str ) );
      }
      catch( const std::exception& )
      {
        // Skip malformed values
      }
    }

    if( !values.empty() )
    {
      m_descriptor_index[uid] = std::move( values );
    }
  }

  LOG_INFO( m_logger, "Loaded " << m_descriptor_index.size() << " descriptors from CSV" );
}


#ifdef VIAME_ENABLE_CPPDB
// -----------------------------------------------------------------------------
void
train_svm_models_process::priv
::load_descriptor_index_from_db()
{
  ::cppdb::session conn( m_conn_str );

  // Query all descriptors from the DESCRIPTOR table
  ::cppdb::result res = conn.create_statement(
    "SELECT UID, VECTOR_DATA FROM DESCRIPTOR" ).query();

  while( res.next() )
  {
    std::string uid;
    std::string vector_data;

    res.fetch( 0, uid );
    res.fetch( 1, vector_data );

    // Parse PostgreSQL array format: {val1,val2,val3,...}
    // Remove curly braces if present
    if( !vector_data.empty() && vector_data.front() == '{' )
    {
      vector_data = vector_data.substr( 1 );
    }
    if( !vector_data.empty() && vector_data.back() == '}' )
    {
      vector_data.pop_back();
    }

    std::vector< double > values;
    std::istringstream ss( vector_data );
    std::string value_str;
    while( std::getline( ss, value_str, ',' ) )
    {
      try
      {
        values.push_back( std::stod( value_str ) );
      }
      catch( const std::exception& )
      {
        // Skip malformed values
      }
    }

    if( !values.empty() )
    {
      m_descriptor_index[uid] = std::move( values );
    }
  }

  conn.close();
  LOG_INFO( m_logger, "Loaded " << m_descriptor_index.size() << " descriptors from database" );
}
#endif


// -----------------------------------------------------------------------------
std::unordered_set< std::string >
train_svm_models_process::priv
::load_uid_file( const std::string& filepath )
{
  std::unordered_set< std::string > uids;
  std::ifstream file( filepath );

  if( !file.is_open() )
  {
    LOG_WARN( m_logger, "Failed to open UID file: " << filepath );
    return uids;
  }

  std::string line;
  while( std::getline( file, line ) )
  {
    while( !line.empty() && ( line.back() == ' ' || line.back() == '\r' || line.back() == '\n' ) )
    {
      line.pop_back();
    }
    if( !line.empty() )
    {
      uids.insert( line );
    }
  }

  return uids;
}


// -----------------------------------------------------------------------------
bool
train_svm_models_process::priv
::directory_exists( const std::string& path )
{
  struct stat info;
  if( stat( path.c_str(), &info ) != 0 )
  {
    return false;
  }
  return ( info.st_mode & S_IFDIR ) != 0;
}


// -----------------------------------------------------------------------------
bool
train_svm_models_process::priv
::create_directory( const std::string& path )
{
  return MKDIR( path.c_str() ) == 0 || directory_exists( path );
}


// -----------------------------------------------------------------------------
std::vector< std::string >
train_svm_models_process::priv
::list_files( const std::string& folder, const std::string& extension )
{
  std::vector< std::string > files;

#if defined(_WIN32) || defined(_WIN64)
  std::string cmd = "dir /b \"" + folder + "\\*." + extension + "\" 2>nul";
#else
  std::string cmd = "ls \"" + folder + "\"/*." + extension + " 2>/dev/null";
#endif

  FILE* pipe = POPEN( cmd.c_str(), "r" );
  if( pipe )
  {
    char buffer[1024];
    while( fgets( buffer, sizeof( buffer ), pipe ) != nullptr )
    {
      std::string file( buffer );
      while( !file.empty() && ( file.back() == '\n' || file.back() == '\r' ) )
      {
        file.pop_back();
      }
      if( !file.empty() )
      {
#if defined(_WIN32) || defined(_WIN64)
        files.push_back( folder + "\\" + file );
#else
        if( file[0] == '/' )
        {
          files.push_back( file );
        }
        else
        {
          files.push_back( folder + "/" + file );
        }
#endif
      }
    }
    PCLOSE( pipe );
  }

  return files;
}


// -----------------------------------------------------------------------------
int
train_svm_models_process::priv
::get_kernel_type() const
{
  if( m_svm_kernel_type == "linear" ) return LINEAR;
  if( m_svm_kernel_type == "poly" ) return POLY;
  if( m_svm_kernel_type == "sigmoid" ) return SIGMOID;
  return RBF;
}


// -----------------------------------------------------------------------------
double
train_svm_models_process::priv
::euclidean_distance( const std::vector< double >& a, const std::vector< double >& b )
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


// -----------------------------------------------------------------------------
double
train_svm_models_process::priv
::cosine_distance( const std::vector< double >& a, const std::vector< double >& b )
{
  if( a.size() != b.size() )
  {
    return std::numeric_limits< double >::max();
  }

  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;

  for( size_t i = 0; i < a.size(); ++i )
  {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  if( norm_a == 0.0 || norm_b == 0.0 )
  {
    return 1.0; // Maximum distance for zero vectors
  }

  double cosine_similarity = dot / ( std::sqrt( norm_a ) * std::sqrt( norm_b ) );
  return 1.0 - cosine_similarity; // Convert similarity to distance
}


// -----------------------------------------------------------------------------
double
train_svm_models_process::priv
::compute_distance( const std::vector< double >& a, const std::vector< double >& b ) const
{
  if( m_distance_metric == "cosine" )
  {
    return cosine_distance( a, b );
  }
  return euclidean_distance( a, b );
}


// -----------------------------------------------------------------------------
std::mt19937
train_svm_models_process::priv
::get_random_generator() const
{
  if( m_random_seed >= 0 )
  {
    return std::mt19937( static_cast< unsigned >( m_random_seed ) );
  }
  std::random_device rd;
  return std::mt19937( rd() );
}


// -----------------------------------------------------------------------------
std::vector< double >
train_svm_models_process::priv
::normalize_vector( const std::vector< double >& v )
{
  double norm = 0.0;
  for( double val : v )
  {
    norm += val * val;
  }
  norm = std::sqrt( norm );

  if( norm == 0.0 )
  {
    return v;
  }

  std::vector< double > result( v.size() );
  for( size_t i = 0; i < v.size(); ++i )
  {
    result[i] = v[i] / norm;
  }
  return result;
}


// -----------------------------------------------------------------------------
bool
train_svm_models_process::priv
::load_npy_vector( const std::string& filepath, std::vector< double >& out )
{
  // Simple NumPy .npy file loader for 1D float64 arrays
  std::ifstream file( filepath, std::ios::binary );
  if( !file.is_open() )
  {
    LOG_ERROR( m_logger, "Failed to open numpy file: " << filepath );
    return false;
  }

  // Read magic string and version
  char magic[6];
  file.read( magic, 6 );
  if( std::strncmp( magic, "\x93NUMPY", 6 ) != 0 )
  {
    LOG_ERROR( m_logger, "Invalid numpy file magic: " << filepath );
    return false;
  }

  uint8_t major_version, minor_version;
  file.read( reinterpret_cast< char* >( &major_version ), 1 );
  file.read( reinterpret_cast< char* >( &minor_version ), 1 );

  // Read header length
  uint32_t header_len = 0;
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

  // Read and parse header
  std::string header( header_len, '\0' );
  file.read( &header[0], header_len );

  // Simple shape parsing
  size_t shape_pos = header.find( "'shape'" );
  if( shape_pos == std::string::npos )
  {
    shape_pos = header.find( "\"shape\"" );
  }
  if( shape_pos == std::string::npos )
  {
    LOG_ERROR( m_logger, "Could not find shape in numpy header: " << filepath );
    return false;
  }

  size_t paren_start = header.find( '(', shape_pos );
  size_t paren_end = header.find( ')', shape_pos );
  if( paren_start == std::string::npos || paren_end == std::string::npos )
  {
    LOG_ERROR( m_logger, "Could not parse shape in numpy header: " << filepath );
    return false;
  }

  std::string shape_str = header.substr( paren_start + 1, paren_end - paren_start - 1 );
  size_t comma_pos = shape_str.find( ',' );
  if( comma_pos != std::string::npos )
  {
    shape_str = shape_str.substr( 0, comma_pos );
  }
  while( !shape_str.empty() && std::isspace( shape_str.front() ) ) shape_str.erase( 0, 1 );
  while( !shape_str.empty() && std::isspace( shape_str.back() ) ) shape_str.pop_back();

  size_t n_elements = std::stoull( shape_str );

  out.resize( n_elements );
  file.read( reinterpret_cast< char* >( out.data() ), n_elements * sizeof( double ) );

  return true;
}


// -----------------------------------------------------------------------------
bool
train_svm_models_process::priv
::load_npy_matrix( const std::string& filepath, std::vector< std::vector< double > >& out )
{
  // Simple NumPy .npy file loader for 2D float64 arrays
  std::ifstream file( filepath, std::ios::binary );
  if( !file.is_open() )
  {
    LOG_ERROR( m_logger, "Failed to open numpy file: " << filepath );
    return false;
  }

  // Read magic string and version
  char magic[6];
  file.read( magic, 6 );
  if( std::strncmp( magic, "\x93NUMPY", 6 ) != 0 )
  {
    LOG_ERROR( m_logger, "Invalid numpy file magic: " << filepath );
    return false;
  }

  uint8_t major_version, minor_version;
  file.read( reinterpret_cast< char* >( &major_version ), 1 );
  file.read( reinterpret_cast< char* >( &minor_version ), 1 );

  // Read header length
  uint32_t header_len = 0;
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

  // Read and parse header
  std::string header( header_len, '\0' );
  file.read( &header[0], header_len );

  // Parse shape: (rows, cols)
  size_t shape_pos = header.find( "'shape'" );
  if( shape_pos == std::string::npos )
  {
    shape_pos = header.find( "\"shape\"" );
  }
  if( shape_pos == std::string::npos )
  {
    LOG_ERROR( m_logger, "Could not find shape in numpy header: " << filepath );
    return false;
  }

  size_t paren_start = header.find( '(', shape_pos );
  size_t paren_end = header.find( ')', shape_pos );
  if( paren_start == std::string::npos || paren_end == std::string::npos )
  {
    LOG_ERROR( m_logger, "Could not parse shape in numpy header: " << filepath );
    return false;
  }

  std::string shape_str = header.substr( paren_start + 1, paren_end - paren_start - 1 );
  size_t comma_pos = shape_str.find( ',' );
  if( comma_pos == std::string::npos )
  {
    LOG_ERROR( m_logger, "Expected 2D shape in numpy file: " << filepath );
    return false;
  }

  std::string rows_str = shape_str.substr( 0, comma_pos );
  std::string cols_str = shape_str.substr( comma_pos + 1 );
  while( !rows_str.empty() && std::isspace( rows_str.front() ) ) rows_str.erase( 0, 1 );
  while( !rows_str.empty() && std::isspace( rows_str.back() ) ) rows_str.pop_back();
  while( !cols_str.empty() && std::isspace( cols_str.front() ) ) cols_str.erase( 0, 1 );
  while( !cols_str.empty() && std::isspace( cols_str.back() ) ) cols_str.pop_back();

  size_t rows = std::stoull( rows_str );
  size_t cols = std::stoull( cols_str );

  // Check for Fortran order
  bool fortran_order = ( header.find( "'fortran_order': True" ) != std::string::npos ) ||
                       ( header.find( "\"fortran_order\": True" ) != std::string::npos );

  // Read data
  std::vector< double > flat_data( rows * cols );
  file.read( reinterpret_cast< char* >( flat_data.data() ), rows * cols * sizeof( double ) );

  // Reshape
  out.resize( rows );
  for( size_t i = 0; i < rows; ++i )
  {
    out[i].resize( cols );
    for( size_t j = 0; j < cols; ++j )
    {
      if( fortran_order )
      {
        out[i][j] = flat_data[j * rows + i];
      }
      else
      {
        out[i][j] = flat_data[i * cols + j];
      }
    }
  }

  return true;
}


// -----------------------------------------------------------------------------
bool
train_svm_models_process::priv
::load_itq_model()
{
  if( m_lsh_model_dir.empty() )
  {
    LOG_ERROR( m_logger, "LSH model directory not specified" );
    return false;
  }

  // Construct model file paths
  std::ostringstream suffix;
  suffix << "b" << m_lsh_bit_length << "_i" << m_lsh_itq_iterations << "_r" << m_lsh_random_seed;

  std::string mean_path = m_lsh_model_dir + "/itq.model." + suffix.str() + ".mean_vec.npy";
  std::string rotation_path = m_lsh_model_dir + "/itq.model." + suffix.str() + ".rotation.npy";

  // Check if files exist
  struct stat info;
  if( stat( mean_path.c_str(), &info ) != 0 )
  {
    LOG_ERROR( m_logger, "ITQ mean vector file not found: " << mean_path );
    return false;
  }
  if( stat( rotation_path.c_str(), &info ) != 0 )
  {
    LOG_ERROR( m_logger, "ITQ rotation matrix file not found: " << rotation_path );
    return false;
  }

  // Load mean vector
  if( !load_npy_vector( mean_path, m_lsh_mean_vec ) )
  {
    return false;
  }
  LOG_INFO( m_logger, "Loaded ITQ mean vector: " << m_lsh_mean_vec.size() << " dimensions" );

  // Load rotation matrix
  if( !load_npy_matrix( rotation_path, m_lsh_rotation ) )
  {
    return false;
  }
  LOG_INFO( m_logger, "Loaded ITQ rotation matrix: " << m_lsh_rotation.size() << " x "
            << ( m_lsh_rotation.empty() ? 0 : m_lsh_rotation[0].size() ) );

  return true;
}


// -----------------------------------------------------------------------------
bool
train_svm_models_process::priv
::load_npy_hash_matrix( const std::string& filepath,
  std::vector< std::vector< bool > >& out, size_t& n_rows, size_t& n_cols )
{
  // Load uint8 numpy array and convert to bool vectors
  std::ifstream file( filepath, std::ios::binary );
  if( !file.is_open() )
  {
    return false;
  }

  // Read magic and version
  char magic[6];
  file.read( magic, 6 );
  if( std::strncmp( magic, "\x93NUMPY", 6 ) != 0 )
  {
    return false;
  }

  uint8_t major_version, minor_version;
  file.read( reinterpret_cast< char* >( &major_version ), 1 );
  file.read( reinterpret_cast< char* >( &minor_version ), 1 );

  uint32_t header_len = 0;
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

  // Parse shape
  size_t shape_pos = header.find( "'shape'" );
  if( shape_pos == std::string::npos )
  {
    shape_pos = header.find( "\"shape\"" );
  }
  if( shape_pos == std::string::npos )
  {
    return false;
  }

  size_t paren_start = header.find( '(', shape_pos );
  size_t paren_end = header.find( ')', shape_pos );
  if( paren_start == std::string::npos || paren_end == std::string::npos )
  {
    return false;
  }

  std::string shape_str = header.substr( paren_start + 1, paren_end - paren_start - 1 );
  size_t comma_pos = shape_str.find( ',' );
  if( comma_pos == std::string::npos )
  {
    return false;
  }

  std::string rows_str = shape_str.substr( 0, comma_pos );
  std::string cols_str = shape_str.substr( comma_pos + 1 );
  while( !rows_str.empty() && std::isspace( rows_str.front() ) ) rows_str.erase( 0, 1 );
  while( !rows_str.empty() && std::isspace( rows_str.back() ) ) rows_str.pop_back();
  while( !cols_str.empty() && std::isspace( cols_str.front() ) ) cols_str.erase( 0, 1 );
  while( !cols_str.empty() && std::isspace( cols_str.back() ) ) cols_str.pop_back();

  n_rows = std::stoull( rows_str );
  n_cols = std::stoull( cols_str );

  // Read uint8 data
  std::vector< uint8_t > flat_data( n_rows * n_cols );
  file.read( reinterpret_cast< char* >( flat_data.data() ), n_rows * n_cols );

  // Convert to bool vectors
  out.resize( n_rows );
  for( size_t i = 0; i < n_rows; ++i )
  {
    out[i].resize( n_cols );
    for( size_t j = 0; j < n_cols; ++j )
    {
      out[i][j] = ( flat_data[i * n_cols + j] != 0 );
    }
  }

  return true;
}


// -----------------------------------------------------------------------------
bool
train_svm_models_process::priv
::load_precomputed_hashes()
{
  if( m_lsh_model_dir.empty() )
  {
    return false;
  }

  std::string hash_codes_path = m_lsh_model_dir + "/lsh_hash_codes.npy";
  std::string hash_uids_path = m_lsh_model_dir + "/lsh_hash_uids.txt";

  // Check if files exist
  struct stat info;
  if( stat( hash_codes_path.c_str(), &info ) != 0 ||
      stat( hash_uids_path.c_str(), &info ) != 0 )
  {
    LOG_INFO( m_logger, "Precomputed hash files not found, will compute at runtime" );
    return false;
  }

  LOG_INFO( m_logger, "Loading precomputed hash codes from: " << hash_codes_path );

  // Load hash codes
  std::vector< std::vector< bool > > hash_codes;
  size_t n_rows, n_cols;
  if( !load_npy_hash_matrix( hash_codes_path, hash_codes, n_rows, n_cols ) )
  {
    LOG_WARN( m_logger, "Failed to load precomputed hash codes" );
    return false;
  }

  // Load UIDs
  std::vector< std::string > uids;
  std::ifstream uid_file( hash_uids_path );
  if( !uid_file.is_open() )
  {
    LOG_WARN( m_logger, "Failed to open UIDs file: " << hash_uids_path );
    return false;
  }

  std::string line;
  while( std::getline( uid_file, line ) )
  {
    // Trim whitespace
    while( !line.empty() && std::isspace( line.back() ) ) line.pop_back();
    while( !line.empty() && std::isspace( line.front() ) ) line.erase( 0, 1 );
    if( !line.empty() )
    {
      uids.push_back( line );
    }
  }

  if( uids.size() != hash_codes.size() )
  {
    LOG_WARN( m_logger, "UID count (" << uids.size() << ") doesn't match hash count ("
              << hash_codes.size() << ")" );
    return false;
  }

  // Build hash index from precomputed data
  m_lsh_hash_codes.clear();
  m_hash_to_uids.clear();

  for( size_t i = 0; i < uids.size(); ++i )
  {
    // Only include UIDs that are in our descriptor index
    if( m_descriptor_index.find( uids[i] ) != m_descriptor_index.end() )
    {
      m_lsh_hash_codes[uids[i]] = hash_codes[i];
      m_hash_to_uids[hash_codes[i]].push_back( uids[i] );
    }
  }

  LOG_INFO( m_logger, "Loaded " << m_lsh_hash_codes.size() << " precomputed hash codes" );
  return true;
}


// -----------------------------------------------------------------------------
void
train_svm_models_process::priv
::initialize_lsh_index()
{
  if( m_lsh_initialized )
  {
    return;
  }

  if( m_nn_index_type != "lsh" )
  {
    return;
  }

  // Try to load precomputed hashes first
  if( load_precomputed_hashes() )
  {
    // Still need ITQ model for computing hashes of query descriptors
    if( !load_itq_model() )
    {
      LOG_WARN( m_logger, "Failed to load ITQ model for query hashing" );
    }
    LOG_INFO( m_logger, "LSH index built from precomputed hashes with "
              << m_hash_to_uids.size() << " unique hash codes" );
    m_lsh_initialized = true;
    return;
  }

  // Fall back to computing hashes at runtime
  if( !load_itq_model() )
  {
    LOG_WARN( m_logger, "Failed to load ITQ model, falling back to brute force NN" );
    m_nn_index_type = "brute_force";
    return;
  }

  LOG_INFO( m_logger, "Computing LSH hash codes for " << m_descriptor_index.size() << " descriptors" );

  m_lsh_hash_codes.clear();
  m_hash_to_uids.clear();

  for( const auto& entry : m_descriptor_index )
  {
    std::vector< bool > hash = compute_hash( entry.second );
    m_lsh_hash_codes[entry.first] = hash;
    m_hash_to_uids[hash].push_back( entry.first );
  }

  LOG_INFO( m_logger, "LSH index built with " << m_hash_to_uids.size() << " unique hash codes" );
  m_lsh_initialized = true;
}


// -----------------------------------------------------------------------------
std::vector< bool >
train_svm_models_process::priv
::compute_hash( const std::vector< double >& descriptor ) const
{
  // Apply ITQ: hash = sign( (normalize(x) - mean) @ rotation )
  std::vector< double > centered;
  if( m_normalize_descriptors )
  {
    std::vector< double > normalized = normalize_vector( descriptor );
    centered.resize( normalized.size() );
    for( size_t i = 0; i < normalized.size() && i < m_lsh_mean_vec.size(); ++i )
    {
      centered[i] = normalized[i] - m_lsh_mean_vec[i];
    }
  }
  else
  {
    centered.resize( descriptor.size() );
    for( size_t i = 0; i < descriptor.size() && i < m_lsh_mean_vec.size(); ++i )
    {
      centered[i] = descriptor[i] - m_lsh_mean_vec[i];
    }
  }

  size_t bit_length = m_lsh_rotation.empty() ? 0 : m_lsh_rotation[0].size();
  std::vector< bool > hash( bit_length );

  for( size_t j = 0; j < bit_length; ++j )
  {
    double sum = 0.0;
    for( size_t i = 0; i < centered.size() && i < m_lsh_rotation.size(); ++i )
    {
      sum += centered[i] * m_lsh_rotation[i][j];
    }
    hash[j] = ( sum >= 0 );
  }

  return hash;
}


// -----------------------------------------------------------------------------
unsigned
train_svm_models_process::priv
::hamming_distance( const std::vector< bool >& a, const std::vector< bool >& b )
{
  unsigned dist = 0;
  size_t n = std::min( a.size(), b.size() );
  for( size_t i = 0; i < n; ++i )
  {
    if( a[i] != b[i] )
    {
      ++dist;
    }
  }
  dist += static_cast< unsigned >( std::max( a.size(), b.size() ) - n );
  return dist;
}


// -----------------------------------------------------------------------------
std::vector< std::pair< std::string, double > >
train_svm_models_process::priv
::find_nearest_neighbors_lsh(
  const std::vector< double >& query,
  size_t k ) const
{
  // Compute hash of query
  std::vector< bool > query_hash = compute_hash( query );

  // Compute hamming distances to all unique hash codes
  using pair_type = std::pair< unsigned, std::vector< bool > >;
  std::priority_queue< pair_type, std::vector< pair_type >, std::less< pair_type > > pq;

  size_t max_hash_candidates = static_cast< size_t >(
    m_hash_to_uids.size() * m_lsh_hash_ratio );
  max_hash_candidates = std::max( max_hash_candidates, k * 10 );

  for( const auto& entry : m_hash_to_uids )
  {
    unsigned dist = hamming_distance( query_hash, entry.first );

    if( pq.size() < max_hash_candidates )
    {
      pq.push( { dist, entry.first } );
    }
    else if( dist < pq.top().first )
    {
      pq.pop();
      pq.push( { dist, entry.first } );
    }
  }

  // Collect candidate UIDs from nearest hash codes
  std::unordered_set< std::string > candidates;
  while( !pq.empty() )
  {
    const auto& uids = m_hash_to_uids.at( pq.top().second );
    candidates.insert( uids.begin(), uids.end() );
    pq.pop();
  }

  // Refine using actual distances
  using dist_pair = std::pair< double, std::string >;
  std::priority_queue< dist_pair > refine_pq;

  for( const auto& uid : candidates )
  {
    auto it = m_descriptor_index.find( uid );
    if( it == m_descriptor_index.end() )
    {
      continue;
    }

    double dist = compute_distance( query, it->second );

    if( refine_pq.size() < k )
    {
      refine_pq.push( { dist, uid } );
    }
    else if( dist < refine_pq.top().first )
    {
      refine_pq.pop();
      refine_pq.push( { dist, uid } );
    }
  }

  // Build result
  std::vector< std::pair< std::string, double > > result;
  result.reserve( refine_pq.size() );
  while( !refine_pq.empty() )
  {
    result.emplace_back( refine_pq.top().second, refine_pq.top().first );
    refine_pq.pop();
  }

  std::reverse( result.begin(), result.end() );
  return result;
}


// -----------------------------------------------------------------------------
std::vector< std::pair< std::string, double > >
train_svm_models_process::priv
::find_nearest_neighbors_lsh_multi(
  const std::vector< std::pair< std::string, std::vector< double > > >& queries,
  size_t k_per_query ) const
{
  std::unordered_map< std::string, double > best_distances;

  for( const auto& query : queries )
  {
    auto neighbors = find_nearest_neighbors_lsh( query.second, k_per_query );
    for( const auto& neighbor : neighbors )
    {
      auto it = best_distances.find( neighbor.first );
      if( it == best_distances.end() || neighbor.second < it->second )
      {
        best_distances[neighbor.first] = neighbor.second;
      }
    }
  }

  std::vector< std::pair< std::string, double > > result;
  result.reserve( best_distances.size() );
  for( const auto& entry : best_distances )
  {
    result.emplace_back( entry.first, entry.second );
  }

  std::sort( result.begin(), result.end(),
    []( const auto& a, const auto& b ) { return a.second < b.second; } );

  return result;
}


// -----------------------------------------------------------------------------
std::vector< std::pair< std::string, double > >
train_svm_models_process::priv
::find_nearest_neighbors( const std::vector< double >& query, size_t k ) const
{
  using pair_type = std::pair< double, std::string >;
  std::priority_queue< pair_type > pq;

  for( const auto& entry : m_descriptor_index )
  {
    double dist = compute_distance( query, entry.second );

    if( pq.size() < k )
    {
      pq.push( { dist, entry.first } );
    }
    else if( dist < pq.top().first )
    {
      pq.pop();
      pq.push( { dist, entry.first } );
    }
  }

  std::vector< std::pair< std::string, double > > result;
  result.reserve( pq.size() );
  while( !pq.empty() )
  {
    result.emplace_back( pq.top().second, pq.top().first );
    pq.pop();
  }

  std::reverse( result.begin(), result.end() );
  return result;
}


// -----------------------------------------------------------------------------
std::vector< std::pair< std::string, double > >
train_svm_models_process::priv
::find_nearest_neighbors_multi(
  const std::vector< std::pair< std::string, std::vector< double > > >& queries,
  size_t k_per_query ) const
{
  // Use LSH if available and configured
  if( m_nn_index_type == "lsh" && m_lsh_initialized )
  {
    return find_nearest_neighbors_lsh_multi( queries, k_per_query );
  }

  // Fall back to brute force
  std::unordered_map< std::string, double > best_distances;

  for( const auto& query : queries )
  {
    auto neighbors = find_nearest_neighbors( query.second, k_per_query );
    for( const auto& neighbor : neighbors )
    {
      auto it = best_distances.find( neighbor.first );
      if( it == best_distances.end() || neighbor.second < it->second )
      {
        best_distances[neighbor.first] = neighbor.second;
      }
    }
  }

  std::vector< std::pair< std::string, double > > result;
  result.reserve( best_distances.size() );
  for( const auto& entry : best_distances )
  {
    result.emplace_back( entry.first, entry.second );
  }

  std::sort( result.begin(), result.end(),
    []( const auto& a, const auto& b ) { return a.second < b.second; } );

  return result;
}


// -----------------------------------------------------------------------------
svm_model*
train_svm_models_process::priv
::train_svm(
  const std::vector< std::pair< std::string, std::vector< double > > >& pos_descriptors,
  const std::vector< std::pair< std::string, std::vector< double > > >& neg_descriptors )
{
  if( pos_descriptors.empty() || neg_descriptors.empty() )
  {
    return nullptr;
  }

  size_t n_pos = pos_descriptors.size();
  size_t n_neg = neg_descriptors.size();
  size_t n_total = n_pos + n_neg;
  size_t dim = pos_descriptors[0].second.size();

  svm_problem prob;
  prob.l = static_cast< int >( n_total );
  prob.y = new double[n_total];
  prob.x = new svm_node*[n_total];

  size_t idx = 0;
  for( const auto& p : pos_descriptors )
  {
    prob.y[idx] = 1.0;
    prob.x[idx] = new svm_node[dim + 1];

    // Optionally normalize the descriptor
    std::vector< double > vec = m_normalize_descriptors ? normalize_vector( p.second ) : p.second;

    for( size_t i = 0; i < dim; ++i )
    {
      prob.x[idx][i].index = static_cast< int >( i + 1 );
      prob.x[idx][i].value = vec[i];
    }
    prob.x[idx][dim].index = -1;
    prob.x[idx][dim].value = 0;
    ++idx;
  }

  for( const auto& n : neg_descriptors )
  {
    prob.y[idx] = -1.0;
    prob.x[idx] = new svm_node[dim + 1];

    // Optionally normalize the descriptor
    std::vector< double > vec = m_normalize_descriptors ? normalize_vector( n.second ) : n.second;

    for( size_t i = 0; i < vec.size() && i < dim; ++i )
    {
      prob.x[idx][i].index = static_cast< int >( i + 1 );
      prob.x[idx][i].value = vec[i];
    }
    prob.x[idx][dim].index = -1;
    prob.x[idx][dim].value = 0;
    ++idx;
  }

  svm_parameter param;
  std::memset( &param, 0, sizeof( param ) );
  param.svm_type = C_SVC;
  param.kernel_type = get_kernel_type();
  param.degree = 3;
  param.gamma = m_svm_gamma;
  param.coef0 = 0;
  param.cache_size = 200;
  param.eps = 0.001;
  param.C = m_svm_c;
  param.nu = 0.5;
  param.p = 0.1;
  param.shrinking = 1;
  // Only enable probability estimates if using "probability" score normalization
  param.probability = ( m_score_normalization == "probability" ) ? 1 : 0;

  // Set up class weights if enabled
  int* weight_labels = nullptr;
  double* weights = nullptr;

  if( m_use_class_weights && n_pos > 0 && n_neg > 0 )
  {
    // Weight classes inversely proportional to their frequency
    double total = static_cast< double >( n_total );
    double pos_weight = total / ( 2.0 * n_pos );
    double neg_weight = total / ( 2.0 * n_neg );

    weight_labels = new int[2];
    weights = new double[2];

    weight_labels[0] = 1;   // Positive class
    weight_labels[1] = -1;  // Negative class
    weights[0] = pos_weight;
    weights[1] = neg_weight;

    param.nr_weight = 2;
    param.weight_label = weight_labels;
    param.weight = weights;

    LOG_DEBUG( m_logger, "Using class weights: positive=" << pos_weight
               << ", negative=" << neg_weight );
  }
  else
  {
    param.nr_weight = 0;
    param.weight_label = nullptr;
    param.weight = nullptr;
  }

  const char* error = svm_check_parameter( &prob, &param );
  svm_model* model = nullptr;

  if( !error )
  {
    svm_set_print_string_function( []( const char* ) {} );
    model = svm_train( &prob, &param );
  }
  else
  {
    LOG_ERROR( m_logger, "SVM parameter check failed: " << error );
  }

  // Clean up training data and weights
  for( size_t i = 0; i < n_total; ++i )
  {
    delete[] prob.x[i];
  }
  delete[] prob.x;
  delete[] prob.y;
  delete[] weight_labels;
  delete[] weights;

  return model;
}


// -----------------------------------------------------------------------------
std::vector< std::pair< std::string, double > >
train_svm_models_process::priv
::score_with_svm(
  svm_model* model,
  const std::unordered_set< std::string >& uids_to_score,
  const std::unordered_set< std::string >& exclude_uids ) const
{
  std::vector< std::pair< std::string, double > > results;

  if( !model )
  {
    return results;
  }

  // Get label order to know which probability/decision value is positive
  int labels[2];
  svm_get_labels( model, labels );
  bool first_is_positive = ( labels[0] == 1 );

  bool use_probability = ( m_score_normalization == "probability" );

  for( const auto& uid : uids_to_score )
  {
    if( exclude_uids.find( uid ) != exclude_uids.end() )
    {
      continue;
    }

    auto it = m_descriptor_index.find( uid );
    if( it == m_descriptor_index.end() )
    {
      continue;
    }

    // Optionally normalize the descriptor
    std::vector< double > vec = m_normalize_descriptors ? normalize_vector( it->second ) : it->second;
    size_t dim = vec.size();

    svm_node* nodes = new svm_node[dim + 1];
    for( size_t i = 0; i < dim; ++i )
    {
      nodes[i].index = static_cast< int >( i + 1 );
      nodes[i].value = vec[i];
    }
    nodes[dim].index = -1;
    nodes[dim].value = 0;

    double positive_score;

    if( use_probability )
    {
      // Use libsvm's probability estimates
      double prob_estimates[2];
      svm_predict_probability( model, nodes, prob_estimates );
      positive_score = first_is_positive ? prob_estimates[0] : prob_estimates[1];
    }
    else
    {
      // Use sigmoid normalization of decision value
      double decision_values[1];
      svm_predict_values( model, nodes, decision_values );
      // Sigmoid: 1.0 / (1.0 + exp(-x))
      // If first label is positive, positive decision value means positive class
      double dv = first_is_positive ? decision_values[0] : -decision_values[0];
      positive_score = 1.0 / ( 1.0 + std::exp( -dv ) );
    }

    results.emplace_back( uid, positive_score );

    delete[] nodes;
  }

  std::sort( results.begin(), results.end(),
    []( const auto& a, const auto& b ) { return a.second > b.second; } );

  return results;
}


// -----------------------------------------------------------------------------
std::vector< std::pair< std::string, double > >
train_svm_models_process::priv
::score_by_centroid_similarity(
  const std::vector< std::pair< std::string, std::vector< double > > >& pos_descriptors,
  const std::unordered_set< std::string >& uids_to_score,
  const std::unordered_set< std::string >& exclude_uids ) const
{
  std::vector< std::pair< std::string, double > > results;

  if( pos_descriptors.empty() )
  {
    return results;
  }

  size_t dim = pos_descriptors[0].second.size();
  std::vector< double > centroid( dim, 0.0 );

  for( const auto& p : pos_descriptors )
  {
    for( size_t i = 0; i < dim && i < p.second.size(); ++i )
    {
      centroid[i] += p.second[i];
    }
  }
  for( size_t i = 0; i < dim; ++i )
  {
    centroid[i] /= pos_descriptors.size();
  }

  for( const auto& uid : uids_to_score )
  {
    if( exclude_uids.find( uid ) != exclude_uids.end() )
    {
      continue;
    }

    auto it = m_descriptor_index.find( uid );
    if( it == m_descriptor_index.end() )
    {
      continue;
    }

    double dist = compute_distance( centroid, it->second );
    double similarity = 1.0 / ( 1.0 + dist );
    results.emplace_back( uid, similarity );
  }

  std::sort( results.begin(), results.end(),
    []( const auto& a, const auto& b ) { return a.second > b.second; } );

  return results;
}


// -----------------------------------------------------------------------------
void
train_svm_models_process::priv
::train_svm_model(
  const std::string& category,
  const std::unordered_set< std::string >& positive_uids,
  const std::unordered_set< std::string >& negative_uids,
  const std::string& output_file )
{
  if( positive_uids.empty() )
  {
    LOG_ERROR( m_logger, "No positive samples for category: " << category );
    return;
  }

  if( negative_uids.empty() )
  {
    LOG_ERROR( m_logger, "No negative samples for category: " << category );
    return;
  }

  // Collect positive descriptors
  std::vector< std::pair< std::string, std::vector< double > > > pos_descriptors;
  for( const auto& uid : positive_uids )
  {
    auto it = m_descriptor_index.find( uid );
    if( it != m_descriptor_index.end() )
    {
      pos_descriptors.emplace_back( uid, it->second );
    }
  }

  if( pos_descriptors.empty() )
  {
    LOG_ERROR( m_logger, "No positive descriptors found for category: " << category );
    return;
  }

  // Check minimum sample threshold
  if( pos_descriptors.size() < m_minimum_positive_threshold )
  {
    LOG_WARN( m_logger, "Category " << category << " has only " << pos_descriptors.size()
              << " positive samples, below threshold of " << m_minimum_positive_threshold
              << ". Skipping." );
    return;
  }

  // Get random generator (seeded or random)
  std::mt19937 gen = get_random_generator();

  // Sample down positives if needed
  if( pos_descriptors.size() > m_maximum_positive_count )
  {
    std::shuffle( pos_descriptors.begin(), pos_descriptors.end(), gen );
    pos_descriptors.resize( m_maximum_positive_count );
  }

  // Compute pos_seed_neighbors if auto mode
  unsigned effective_pos_seed_neighbors = m_pos_seed_neighbors;
  if( m_auto_compute_neighbors && m_maximum_positive_count > 0 )
  {
    effective_pos_seed_neighbors = m_maximum_negative_count / m_maximum_positive_count;
    if( m_train_on_neighbors_only )
    {
      effective_pos_seed_neighbors *= 2;
    }
    if( effective_pos_seed_neighbors < m_min_pos_seed_neighbors )
    {
      effective_pos_seed_neighbors = m_min_pos_seed_neighbors;
    }
  }
  else
  {
    // Even if not auto-computing, still enforce minimum
    if( effective_pos_seed_neighbors < m_min_pos_seed_neighbors )
    {
      effective_pos_seed_neighbors = m_min_pos_seed_neighbors;
    }
  }

  // Determine max negatives for each source
  unsigned max_nn_negatives = m_maximum_negative_count;
  unsigned max_random_negatives = 0;

  if( !m_train_on_neighbors_only )
  {
    max_nn_negatives = m_maximum_negative_count / 2;
    max_random_negatives = m_maximum_negative_count / 2;
  }

  LOG_INFO( m_logger, "Stage 1: Finding working index via nearest neighbors" );

  auto nn_results = find_nearest_neighbors_multi( pos_descriptors, effective_pos_seed_neighbors );

  std::unordered_set< std::string > working_index_uids;
  for( const auto& nn : nn_results )
  {
    if( negative_uids.find( nn.first ) != negative_uids.end() &&
        positive_uids.find( nn.first ) == positive_uids.end() )
    {
      working_index_uids.insert( nn.first );
    }
  }

  LOG_INFO( m_logger, "Working index contains " << working_index_uids.size()
            << " candidate negatives" );

  std::unordered_set< std::string > hard_negative_uids;

  if( m_two_stage_training && !working_index_uids.empty() )
  {
    LOG_INFO( m_logger, "Stage 1: Computing initial ranking" );

    std::unordered_set< std::string > pos_uid_set;
    for( const auto& p : pos_descriptors )
    {
      pos_uid_set.insert( p.first );
    }

    auto ranked_results = score_by_centroid_similarity(
      pos_descriptors, working_index_uids, pos_uid_set );

    for( const auto& result : ranked_results )
    {
      if( hard_negative_uids.size() >= max_nn_negatives )
      {
        break;
      }
      hard_negative_uids.insert( result.first );
    }

    LOG_INFO( m_logger, "Stage 1: Selected " << hard_negative_uids.size()
              << " hard negatives from initial ranking" );

    // If svm_rerank_negatives is enabled, train an intermediate SVM to re-rank
    // Otherwise, just use the centroid-similarity selected negatives
    if( m_svm_rerank_negatives && !hard_negative_uids.empty() )
    {
      std::vector< std::pair< std::string, std::vector< double > > > initial_neg_descriptors;
      for( const auto& uid : hard_negative_uids )
      {
        auto it = m_descriptor_index.find( uid );
        if( it != m_descriptor_index.end() )
        {
          initial_neg_descriptors.emplace_back( uid, it->second );
        }
      }

      svm_model* initial_model = train_svm( pos_descriptors, initial_neg_descriptors );

      if( initial_model )
      {
        LOG_INFO( m_logger, "Stage 2: Re-ranking with initial SVM" );

        auto svm_ranked_results = score_with_svm(
          initial_model, working_index_uids, pos_uid_set );

        hard_negative_uids.clear();
        for( const auto& result : svm_ranked_results )
        {
          if( hard_negative_uids.size() >= max_nn_negatives )
          {
            break;
          }
          hard_negative_uids.insert( result.first );
        }

        LOG_INFO( m_logger, "Stage 2: Selected " << hard_negative_uids.size()
                  << " hard negatives from SVM ranking" );

        // Add feedback samples (near decision boundary) if enabled
        if( m_feedback_sample_count > 0 )
        {
          // Sort by distance from 0.5 (decision boundary)
          std::vector< std::pair< std::string, double > > feedback_candidates;
          for( const auto& result : svm_ranked_results )
          {
            if( hard_negative_uids.find( result.first ) == hard_negative_uids.end() &&
                negative_uids.find( result.first ) != negative_uids.end() )
            {
              double uncertainty = std::abs( result.second - 0.5 );
              feedback_candidates.emplace_back( result.first, uncertainty );
            }
          }

          // Sort by uncertainty (lowest first = closest to decision boundary)
          std::sort( feedback_candidates.begin(), feedback_candidates.end(),
            []( const auto& a, const auto& b ) { return a.second < b.second; } );

          // Add most uncertain samples as additional hard negatives
          unsigned added = 0;
          for( const auto& candidate : feedback_candidates )
          {
            if( added >= m_feedback_sample_count )
            {
              break;
            }
            hard_negative_uids.insert( candidate.first );
            ++added;
          }

          LOG_INFO( m_logger, "Added " << added << " feedback samples near decision boundary" );
        }

        svm_free_and_destroy_model( &initial_model );
      }
    }
  }
  else
  {
    for( const auto& uid : working_index_uids )
    {
      if( hard_negative_uids.size() >= max_nn_negatives )
      {
        break;
      }
      hard_negative_uids.insert( uid );
    }
  }

  // Collect final negative descriptors
  std::vector< std::pair< std::string, std::vector< double > > > neg_descriptors;

  for( const auto& uid : hard_negative_uids )
  {
    auto it = m_descriptor_index.find( uid );
    if( it != m_descriptor_index.end() )
    {
      neg_descriptors.emplace_back( uid, it->second );
    }
  }

  // Add random negatives if not train_on_neighbors_only
  if( !m_train_on_neighbors_only && max_random_negatives > 0 )
  {
    std::vector< std::string > remaining_negatives;
    for( const auto& uid : negative_uids )
    {
      if( hard_negative_uids.find( uid ) == hard_negative_uids.end() )
      {
        remaining_negatives.push_back( uid );
      }
    }

    if( !remaining_negatives.empty() )
    {
      std::shuffle( remaining_negatives.begin(), remaining_negatives.end(), gen );

      size_t add_count = std::min(
        static_cast< size_t >( max_random_negatives ),
        remaining_negatives.size() );

      for( size_t i = 0; i < add_count; ++i )
      {
        auto it = m_descriptor_index.find( remaining_negatives[i] );
        if( it != m_descriptor_index.end() )
        {
          neg_descriptors.emplace_back( remaining_negatives[i], it->second );
        }
      }
    }

    LOG_INFO( m_logger, "Added " << ( neg_descriptors.size() - hard_negative_uids.size() )
              << " random negatives" );
  }

  if( neg_descriptors.empty() )
  {
    LOG_ERROR( m_logger, "No negative descriptors found for category: " << category );
    return;
  }

  // Check minimum negative threshold
  if( neg_descriptors.size() < m_minimum_negative_threshold )
  {
    LOG_WARN( m_logger, "Category " << category << " has only " << neg_descriptors.size()
              << " negative samples, below threshold of " << m_minimum_negative_threshold
              << ". Skipping." );
    return;
  }

  LOG_INFO( m_logger, "Final training: " << pos_descriptors.size() << " positive, "
            << neg_descriptors.size() << " negative samples" );

  svm_model* final_model = train_svm( pos_descriptors, neg_descriptors );

  if( final_model )
  {
    if( svm_save_model( output_file.c_str(), final_model ) == 0 )
    {
      LOG_INFO( m_logger, "Saved SVM model to: " << output_file );
    }
    else
    {
      LOG_ERROR( m_logger, "Failed to save SVM model: " << output_file );
    }
    svm_free_and_destroy_model( &final_model );
  }
  else
  {
    LOG_ERROR( m_logger, "SVM training failed for category: " << category );
  }
}


// -----------------------------------------------------------------------------
void
train_svm_models_process::priv
::train_all_models()
{
  load_descriptor_index();

  if( m_descriptor_index.empty() )
  {
    LOG_ERROR( m_logger, "No descriptors loaded from index" );
    return;
  }

  // Initialize LSH index if configured
  initialize_lsh_index();

  if( !directory_exists( m_output_directory ) )
  {
    if( !create_directory( m_output_directory ) )
    {
      LOG_ERROR( m_logger, "Failed to create output directory: " << m_output_directory );
      return;
    }
  }

  auto label_files = list_files( m_label_folder, m_label_extension );

  if( label_files.empty() )
  {
    LOG_ERROR( m_logger, "No label files found in: " << m_label_folder );
    return;
  }

  LOG_INFO( m_logger, "Found " << label_files.size() << " label files" );

  std::vector< std::string > categories;
  std::unordered_map< std::string, std::unordered_set< std::string > > category_uids;
  std::unordered_set< std::string > all_uids;

  for( const auto& filepath : label_files )
  {
    std::string filename;
    size_t pos = filepath.find_last_of( "/\\" );
    if( pos != std::string::npos )
    {
      filename = filepath.substr( pos + 1 );
    }
    else
    {
      filename = filepath;
    }

    pos = filename.find( '.' );
    std::string category = ( pos != std::string::npos ) ? filename.substr( 0, pos ) : filename;

    auto uids = load_uid_file( filepath );
    categories.push_back( category );
    category_uids[category] = uids;
    all_uids.insert( uids.begin(), uids.end() );

    LOG_INFO( m_logger, "Category " << category << ": " << uids.size() << " UIDs" );
  }

  for( const auto& category : categories )
  {
    if( category == m_background_category )
    {
      LOG_INFO( m_logger, "Skipping background category: " << category );
      continue;
    }

    LOG_INFO( m_logger, "Training model for category: " << category );

    const auto& positive_uids = category_uids[category];

    std::unordered_set< std::string > negative_uids;
    for( const auto& uid : all_uids )
    {
      if( positive_uids.find( uid ) == positive_uids.end() )
      {
        negative_uids.insert( uid );
      }
    }

    std::string output_file = m_output_directory + "/" + category + ".svm";
    train_svm_model( category, positive_uids, negative_uids, output_file );
  }

  LOG_INFO( m_logger, "SVM training complete" );
}


// =============================================================================

train_svm_models_process
::train_svm_models_process( config_block_sptr const& config )
  : process( config ),
    d( new train_svm_models_process::priv() )
{
  make_ports();
  make_config();
}


train_svm_models_process
::~train_svm_models_process()
{
}


// -----------------------------------------------------------------------------
void
train_svm_models_process
::_configure()
{
  d->m_descriptor_index_file = config_value_using_trait( descriptor_index_file );
  d->m_conn_str = config_value_using_trait( conn_str );
  d->m_label_folder = config_value_using_trait( label_folder );
  d->m_label_extension = config_value_using_trait( label_extension );
  d->m_output_directory = config_value_using_trait( output_directory );
  d->m_background_category = config_value_using_trait( background_category );
  d->m_maximum_positive_count = config_value_using_trait( maximum_positive_count );
  d->m_maximum_negative_count = config_value_using_trait( maximum_negative_count );
  d->m_minimum_positive_threshold = config_value_using_trait( minimum_positive_threshold );
  d->m_minimum_negative_threshold = config_value_using_trait( minimum_negative_threshold );
  d->m_train_on_neighbors_only = config_value_using_trait( train_on_neighbors_only );
  d->m_two_stage_training = config_value_using_trait( two_stage_training );
  d->m_svm_rerank_negatives = config_value_using_trait( svm_rerank_negatives );
  d->m_auto_compute_neighbors = config_value_using_trait( auto_compute_neighbors );
  d->m_pos_seed_neighbors = config_value_using_trait( pos_seed_neighbors );
  d->m_min_pos_seed_neighbors = config_value_using_trait( min_pos_seed_neighbors );
  d->m_feedback_sample_count = config_value_using_trait( feedback_sample_count );
  d->m_svm_kernel_type = config_value_using_trait( svm_kernel_type );
  d->m_svm_c = config_value_using_trait( svm_c );
  d->m_svm_gamma = config_value_using_trait( svm_gamma );
  d->m_normalize_descriptors = config_value_using_trait( normalize_descriptors );
  d->m_score_normalization = config_value_using_trait( score_normalization );
  d->m_use_class_weights = config_value_using_trait( use_class_weights );
  d->m_distance_metric = config_value_using_trait( distance_metric );
  d->m_random_seed = config_value_using_trait( random_seed );
  d->m_nn_index_type = config_value_using_trait( nn_index_type );
  d->m_lsh_model_dir = config_value_using_trait( lsh_model_dir );
  d->m_lsh_bit_length = config_value_using_trait( lsh_bit_length );
  d->m_lsh_itq_iterations = config_value_using_trait( lsh_itq_iterations );
  d->m_lsh_random_seed = config_value_using_trait( lsh_random_seed );
  d->m_lsh_hash_ratio = config_value_using_trait( lsh_hash_ratio );

  d->m_logger = this->logger();
}


// -----------------------------------------------------------------------------
void
train_svm_models_process
::_step()
{
  if( !d->m_training_complete )
  {
    d->train_all_models();
    d->m_training_complete = true;
  }

  mark_process_as_complete();
}


// -----------------------------------------------------------------------------
void
train_svm_models_process
::make_ports()
{
  // No input/output ports - reads from files and writes to files
}


// -----------------------------------------------------------------------------
void
train_svm_models_process
::make_config()
{
  declare_config_using_trait( descriptor_index_file );
  declare_config_using_trait( conn_str );
  declare_config_using_trait( label_folder );
  declare_config_using_trait( label_extension );
  declare_config_using_trait( output_directory );
  declare_config_using_trait( background_category );
  declare_config_using_trait( maximum_positive_count );
  declare_config_using_trait( maximum_negative_count );
  declare_config_using_trait( minimum_positive_threshold );
  declare_config_using_trait( minimum_negative_threshold );
  declare_config_using_trait( train_on_neighbors_only );
  declare_config_using_trait( two_stage_training );
  declare_config_using_trait( svm_rerank_negatives );
  declare_config_using_trait( auto_compute_neighbors );
  declare_config_using_trait( pos_seed_neighbors );
  declare_config_using_trait( min_pos_seed_neighbors );
  declare_config_using_trait( feedback_sample_count );
  declare_config_using_trait( svm_kernel_type );
  declare_config_using_trait( svm_c );
  declare_config_using_trait( svm_gamma );
  declare_config_using_trait( normalize_descriptors );
  declare_config_using_trait( score_normalization );
  declare_config_using_trait( use_class_weights );
  declare_config_using_trait( distance_metric );
  declare_config_using_trait( random_seed );
  declare_config_using_trait( nn_index_type );
  declare_config_using_trait( lsh_model_dir );
  declare_config_using_trait( lsh_bit_length );
  declare_config_using_trait( lsh_itq_iterations );
  declare_config_using_trait( lsh_random_seed );
  declare_config_using_trait( lsh_hash_ratio );
}

} // end namespace svm

} // end namespace viame
