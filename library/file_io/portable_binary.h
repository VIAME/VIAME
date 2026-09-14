// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief The binary layout of a KWFD feature/descriptor file.
///
/// This is what cereal's `PortableBinaryArchive` did, and only that: P8-T06
/// deletes cereal, and every KWFD file anyone already has was written by it.
/// The format is recorded byte for byte in
/// `library/file_io/tests/test_feature_descriptor_io.cxx`, taken before the
/// swap.
///
/// The whole of it:
///
///  * the first byte of the stream says whether the numbers that follow are
///    little-endian. A reader on the other kind of machine swaps every value
///    as it reads; nothing else about the layout changes;
///  * an arithmetic value is its own bytes, in order, with no tag, no size
///    and no padding;
///  * anything else is its members, in the order its `serialize` names them.
///
/// That last point is why this is an archive rather than a pair of
/// functions. `feature_`, `covariance_`, `rgb_color`, `vector_` and
/// `matrix_` each say what their bytes are, as a `serialize` template over
/// the archive type, and those declarations did not move -- so the layout is
/// still defined where it was and this file cannot disagree with it.

#ifndef VIAME_FILE_IO_PORTABLE_BINARY_H_
#define VIAME_FILE_IO_PORTABLE_BINARY_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace viame {

namespace portable_binary {

// ----------------------------------------------------------------------------
inline bool
host_is_little_endian()
{
  static std::int32_t const test = 1;
  return *reinterpret_cast< std::int8_t const* >( &test ) == 1;
}

// ----------------------------------------------------------------------------
inline void
swap_bytes( std::uint8_t* data, std::size_t size )
{
  for( std::size_t i = 0, end = size / 2; i < end; ++i )
  {
    std::swap( data[ i ], data[ size - i - 1 ] );
  }
}

// ----------------------------------------------------------------------------
/// Does `T` say what its own bytes are?
template < typename T, typename Archive, typename = void >
struct has_member_serialize : std::false_type {};

template < typename T, typename Archive >
struct has_member_serialize<
  T, Archive,
  std::void_t< decltype(
    std::declval< T& >().serialize( std::declval< Archive& >() ) ) > >
  : std::true_type {};

// ----------------------------------------------------------------------------
/// Writes the layout above.
class writer
{
public:
  explicit writer( std::ostream& stream )
    : m_stream( stream ),
      m_swap( false )
  {
    // The flag is a value like any other, so it goes through the same path;
    // it is written before `m_swap` could matter, and a single byte is the
    // same either way round.
    std::uint8_t const little = host_is_little_endian() ? 1 : 0;
    operator()( little );
  }

  template < typename... Args >
  writer&
  operator()( Args&&... args )
  {
    ( one( std::forward< Args >( args ) ), ... );
    return *this;
  }

private:
  template < typename T >
  void
  one( T&& value )
  {
    using bare = std::decay_t< T >;

    // Serialization does not modify what it is given, but the `serialize`
    // members are one template shared by reading and writing and so are not
    // const. cereal did the same.
    bare& item = const_cast< bare& >( static_cast< bare const& >( value ) );

    if constexpr( std::is_arithmetic_v< bare > )
    {
      std::uint8_t bytes[ sizeof( bare ) ];
      std::memcpy( bytes, &item, sizeof( bare ) );
      if( m_swap ) { swap_bytes( bytes, sizeof( bare ) ); }
      m_stream.write(
        reinterpret_cast< char const* >( bytes ),
        static_cast< std::streamsize >( sizeof( bare ) ) );
    }
    else if constexpr( has_member_serialize< bare, writer >::value )
    {
      item.serialize( *this );
    }
    else
    {
      // Found by argument-dependent lookup, which is how the free
      // `serialize` for `vector_` and `matrix_` is reached.
      serialize( *this, item );
    }
  }

  std::ostream& m_stream;
  bool m_swap;
};

// ----------------------------------------------------------------------------
/// Reads what `writer` wrote.
class reader
{
public:
  explicit reader( std::istream& stream )
    : m_stream( stream ),
      m_swap( false )
  {
    std::uint8_t little = 0;
    operator()( little );
    m_swap = ( ( host_is_little_endian() ? 1 : 0 ) != little );
  }

  template < typename... Args >
  reader&
  operator()( Args&... args )
  {
    ( one( args ), ... );
    return *this;
  }

private:
  template < typename T >
  void
  one( T& item )
  {
    if constexpr( std::is_arithmetic_v< T > )
    {
      std::uint8_t bytes[ sizeof( T ) ];
      m_stream.read(
        reinterpret_cast< char* >( bytes ),
        static_cast< std::streamsize >( sizeof( T ) ) );
      if( m_stream.gcount() !=
          static_cast< std::streamsize >( sizeof( T ) ) )
      {
        throw std::runtime_error(
          "unexpected end of a portable binary stream" );
      }
      if( m_swap ) { swap_bytes( bytes, sizeof( T ) ); }
      std::memcpy( &item, bytes, sizeof( T ) );
    }
    else if constexpr( has_member_serialize< T, reader >::value )
    {
      item.serialize( *this );
    }
    else
    {
      serialize( *this, item );
    }
  }

  std::istream& m_stream;
  bool m_swap;
};

} // namespace portable_binary

} // namespace viame

#endif // VIAME_FILE_IO_PORTABLE_BINARY_H_
