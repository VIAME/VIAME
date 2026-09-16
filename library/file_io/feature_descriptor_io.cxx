// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Core feature_descriptor_io implementation

#include "feature_descriptor_io.h"

#include <fstream>

#include "portable_binary.h"
#include <viame/algorithm_framework/exceptions.h>
#include <viame/algorithm_framework/viame_compiler_config.h>

using namespace viame;

namespace viame {

namespace core {

// Private implementation class
class feature_descriptor_io::priv
{
public:
  priv( feature_descriptor_io& parent )
    : parent( parent )
  {}

  feature_descriptor_io& parent;

  // Configuration values
  bool c_write_float_features() { return parent.c_write_float_features; }
};

// -------------------------------------------------------------------------
void
feature_descriptor_io
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.feature_descriptor_io" );
}

// Destructor
feature_descriptor_io
::~feature_descriptor_io()
{}

// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
bool
feature_descriptor_io
::check_configuration( [[maybe_unused]] viame::config_block_sptr config ) const
{
  return true;
}

namespace {

// Helper function to serialized a vector of features of known type
template < typename Archive, typename T >
void
save_features( Archive& ar, std::vector< feature_sptr > const& features )
{
  for( auto const& f : features )
  {
    if( !f )
    {
      VITAL_THROW( viame::invalid_data, "not able to write a Null feature" );
    }
    if( auto ft = std::dynamic_pointer_cast< feature_< T > >( f ) )
    {
      ar( *ft );
    }
    else
    {
      ar( feature_< T >( *f ) );
    }
  }
}

// ----------------------------------------------------------------------------
// Helper function to unserialized a vector of N features of known type
template < typename Archive, typename T >
viame::feature_set_sptr
read_features( Archive& ar, size_t num_feat )
{
  std::vector< feature_sptr > features;
  features.reserve( num_feat );
  for( size_t i = 0; i < num_feat; ++i )
  {
    auto f = std::make_shared< feature_< T > >();
    ar( *f );
    features.push_back( f );
  }
  return std::make_shared< viame::simple_feature_set >( features );
}

// ----------------------------------------------------------------------------
// Helper function to serialized a vector of descriptors of known type
template < typename Archive, typename T >
void
save_descriptors( Archive& ar, descriptor_set_sptr const& descriptors )
{
  // dimensionality of each descriptor
  std::uint64_t dim = descriptors->at( 0 )->size();
  ar( dim );
  for( descriptor_sptr const d : *descriptors )
  {
    if( !d )
    {
      VITAL_THROW( viame::invalid_data, "not able to write a Null descriptor" );
    }
    if( d->size() != dim )
    {
      VITAL_THROW(
        viame::invalid_data, std::string( "descriptor dimension is not " ) +
        "consistent, should be " + std::to_string( dim ) +
        ", is " + std::to_string( d->size() ) );
    }
    if( auto dt = std::dynamic_pointer_cast< descriptor_array_of< T > >( d ) )
    {
      const T* data = dt->raw_data();
      for( size_t i = 0; i < dim; ++i, ++data )
      {
        ar( *data );
      }
    }
    else
    {
      VITAL_THROW(
        viame::invalid_data, std::string( "saving descriptors of type " ) +
        typeid( T ).name() + " but received type " +
        d->data_type().name() );
    }
  }
}

// ----------------------------------------------------------------------------
// Helper function to unserialized a vector of N descriptors of known type
template < typename Archive, typename T >
viame::descriptor_set_sptr
read_descriptors( Archive& ar, size_t num_desc )
{
  // dimensionality of each descriptor
  std::uint64_t dim;
  ar( dim );

  std::vector< descriptor_sptr > descriptors;
  descriptors.reserve( num_desc );
  for( size_t i = 0; i < num_desc; ++i )
  {
    std::shared_ptr< descriptor_array_of< T > > d;
    // allocate fixed vectors for common dimensions
    switch( dim )
    {
      case 128:
        d = std::make_shared< descriptor_fixed< T, 128 > >();
        break;
      case 64:
        d = std::make_shared< descriptor_fixed< T, 64 > >();
        break;
      default:
        d = std::make_shared< descriptor_dynamic< T > >( dim );
    }

    T* data = d->raw_data();
    for( size_t x = 0; x < dim; ++x, ++data )
    {
      ar( *data );
    }
    descriptors.push_back( d );
  }
  return std::make_shared< viame::simple_descriptor_set >( descriptors );
}

// ----------------------------------------------------------------------------
// compute base 2 log of integers at compile time
constexpr size_t
log2( size_t n )
{
  return ( ( n < 2 ) ? 0 : 1 + log2( n / 2 ) );
}

// ----------------------------------------------------------------------------
// compute a unique byte code for built-in types
template < typename T >
struct type_traits
{
  constexpr static uint8_t code = static_cast< uint8_t >(
    ( std::numeric_limits< T >::is_integer << 5 ) +
    ( std::numeric_limits< T >::is_signed << 4 ) +
    log2( sizeof( T ) ) );
};

// ----------------------------------------------------------------------------
uint8_t
code_from_typeid( std::type_info const& tid )
{
#define CODE_TYPE( T )           \
if( tid == typeid( T ) )         \
{                                \
  return type_traits< T >::code; \
}

  CODE_TYPE( uint8_t );
  CODE_TYPE( int8_t );
  CODE_TYPE( uint16_t );
  CODE_TYPE( int16_t );
  CODE_TYPE( uint32_t );
  CODE_TYPE( int32_t );
  CODE_TYPE( uint64_t );
  CODE_TYPE( int64_t );
  CODE_TYPE( float );
  CODE_TYPE( double );

#undef CODE_TYPE
  return 0;
}

} // namespace

// ----------------------------------------------------------------------------
// Implementation specific load functionality.
void
feature_descriptor_io
::load_(
  std::string const& filename,
  viame::feature_set_sptr& feat,
  viame::descriptor_set_sptr& desc ) const
{
  // open input file
  std::ifstream ifile( filename.c_str(), std::ios::binary );

  // read "magic numbers" to validate this file as a KWIVER feature descriptor
  // file
  char file_id[ 5 ] = { 0 };
  ifile.read( file_id, 4 );
  if( std::strncmp( file_id, "KWFD", 4 ) != 0 )
  {
    VITAL_THROW(
      viame::invalid_data,
      "Does not look like a KWIVER feature/descriptor file: " +
      filename );
  }

  typedef viame::portable_binary::reader Archive_t;

  Archive_t ar( ifile );

  // file format version
  uint16_t version;
  ar( version );
  if( version != 1 )
  {
    VITAL_THROW(
      viame::invalid_data, "Unknown file format version: " +
      std::to_string( version ) );
  }

  std::uint64_t num_feat = 0;
  ar( num_feat );
  if( num_feat > 0 )
  {
    uint8_t type_code;
    ar( type_code );
    switch( type_code )
    {
      case type_traits< float >::code:
        feat = read_features< Archive_t, float >( ar, num_feat );
        break;
      case type_traits< double >::code:
        feat = read_features< Archive_t, double >( ar, num_feat );
        break;
      default:
        VITAL_THROW(
          viame::invalid_data, "unknown feature type code: " +
          std::to_string( type_code ) );
    }
  }
  else
  {
    feat = feature_set_sptr();
  }

  std::uint64_t num_desc = 0;
  ar( num_desc );
  if( num_desc > 0 )
  {
    uint8_t type_code;
    ar( type_code );
    switch( type_code )
    {
#define DO_CASE( T )                                         \
  case type_traits< T >::code:                               \
    desc = read_descriptors< Archive_t, T >( ar, num_desc ); \
    break

    DO_CASE( uint8_t );
    DO_CASE( int8_t );
    DO_CASE( uint16_t );
    DO_CASE( int16_t );
    DO_CASE( uint32_t );
    DO_CASE( int32_t );
    DO_CASE( uint64_t );
    DO_CASE( int64_t );
    DO_CASE( float );
    DO_CASE( double );
#undef DO_CASE

      default:
        VITAL_THROW(
          viame::invalid_data, "unknown descriptor type code: " +
          std::to_string( type_code ) );
    }
  }
  else
  {
    desc = descriptor_set_sptr();
  }
}

// ----------------------------------------------------------------------------
// Implementation specific save functionality.
void
feature_descriptor_io
::save_(
  std::string const& filename,
  viame::feature_set_sptr feat,
  viame::descriptor_set_sptr desc ) const
{
  if( !( feat && feat->size() > 0 ) &&
      !( desc && desc->size() > 0 ) )
  {
    LOG_WARN(logger(), "Not writing file, no features or descriptors" );
    return;
  }

  // open output file
  std::ofstream ofile( filename.c_str(), std::ios::binary );
  // write "magic numbers" to identify this file as a KWIVER feature descriptor
  // file
  ofile.write( "KWFD", 4 );

  typedef viame::portable_binary::writer Archive_t;
  Archive_t ar( ofile );

  // file format version
  uint16_t version = 1;
  ar( version );

  if( feat && feat->size() > 0 )
  {
    std::vector< feature_sptr > features = feat->features();

    // number of elements
    ar( static_cast< std::uint64_t >( features.size() ) );
    uint8_t type_code = code_from_typeid( features[ 0 ]->data_type() );
    // if requested, force the output format to use floats instead of doubles
    if( d_->c_write_float_features() )
    {
      type_code = type_traits< float >::code;
    }
    ar( type_code );
    switch( type_code )
    {
      case type_traits< float >::code:
        save_features< Archive_t, float >( ar, features );
        break;
      case type_traits< double >::code:
        save_features< Archive_t, double >( ar, features );
        break;
      default:
        VITAL_THROW( viame::invalid_data, "features must be float or double" );
    }
  }
  else
  {
    ar( static_cast< std::uint64_t >( 0 ) ); // number of elements
  }

  if( desc && desc->size() > 0 )
  {
    // number of elements
    ar( static_cast< std::uint64_t >( desc->size() ) );
    uint8_t type_code = code_from_typeid( desc->at( 0 )->data_type() );
    ar( type_code );
    switch( type_code )
    {
#define DO_CASE( T )                              \
  case type_traits< T >::code:                    \
    save_descriptors< Archive_t, T >( ar, desc ); \
    break

    DO_CASE( uint8_t );
    DO_CASE( int8_t );
    DO_CASE( uint16_t );
    DO_CASE( int16_t );
    DO_CASE( uint32_t );
    DO_CASE( int32_t );
    DO_CASE( uint64_t );
    DO_CASE( int64_t );
    DO_CASE( float );
    DO_CASE( double );
#undef DO_CASE

      default:
        VITAL_THROW(
          viame::invalid_data,
          std::string( "descriptor type not supported: " ) +
          desc->at( 0 )->data_type().name() );
    }
  }
  else
  {
    ar( static_cast< std::uint64_t >( 0 ) ); // number of elements
  }
}

} // namespace core

} // namespace viame
