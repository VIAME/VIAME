// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_BITFLAGS_H
#define KWIVER_VITAL_BITFLAGS_H

#include <type_traits>

#define KWIVER_DECLARE_BITFLAGS( _flags, _enum ) \
  using _flags = ::kwiver::vital::bitflags< _enum >

#define KWIVER_DECLARE_OPERATORS_FOR_BITFLAGS( _flags ) \
  inline _flags operator|( _flags::enum_t f1, _flags::enum_t f2 ) \
    { return _flags{ f1 } | f2; } \
  inline _flags operator|( _flags::enum_t f1, _flags f2 ) \
    { return f2 | f1; } \
  inline ::kwiver::vital::incompatible_flag operator|( \
    _flags::enum_t f1, int f2 ) \
    { return {}; }

namespace kwiver {
namespace vital {

//-----------------------------------------------------------------------------
class incompatible_flag
{
};

//-----------------------------------------------------------------------------
template < typename Enum >
class bitflags
{
public:
  using enum_t = Enum;
  using int_t = typename std::underlying_type< Enum >::type;
  using uint_t = typename std::make_unsigned< int_t >::type;

  inline bitflags() : m_i{ 0 } {}
  inline bitflags( enum_t flag ) : m_i{ static_cast< int_t >( flag ) } {}
  inline bitflags( bitflags const& other) = default;

  inline bitflags& operator=( bitflags const& other ) = default;

  // Bitwise and
  inline bitflags& operator&=( int_t mask )
  { this->m_i &= mask; return *this; }

  inline bitflags& operator&=( uint_t mask )
  { this->m_i &= mask; return *this; }

  inline bitflags operator&( enum_t f ) const
  { bitflags g; g.m_i = this->m_i & static_cast< int_t >( f ); return g; }

  inline bitflags operator&( int_t mask ) const
  { bitflags g; g.m_i = this->m_i & mask; return g; }

  inline bitflags operator&( uint_t mask ) const
  { bitflags g; g.m_i = this->m_i & static_cast< int_t >( mask ); return g; }

  // Bitwise or
  inline bitflags& operator|=( bitflags f )
  { this->m_i |= f.m_i; return *this; }

  inline bitflags& operator|=( enum_t f )
  { this->m_i |= static_cast< int_t >( f ); return *this; }

  inline bitflags operator|( bitflags f ) const
  { bitflags g; g.m_i = this->m_i | f.m_i; return g; }

  inline bitflags operator|( enum_t f ) const
  { bitflags g; g.m_i = this->m_i | static_cast< int_t >( f ); return g; }

  // Bitwise xor
  inline bitflags& operator^=( bitflags f )
  { this->m_i ^= f.m_i; return *this; }

  inline bitflags& operator^=( enum_t f )
  { this->m_i ^= static_cast< int_t >( f ); return *this; }

  inline bitflags operator^( bitflags f ) const
  { bitflags g; g.m_i = this->m_i ^ f.m_i; return g; }

  inline bitflags operator^( enum_t f ) const
  { bitflags g; g.m_i = this->m_i ^ static_cast< int_t >( f ); return g; }

  // Implicit conversion to plain type
  inline operator int_t() const { return this->m_i; }

  // Bitwise invert
  inline int_t operator~() const { return ~this->m_i; }

  // Test if no bits are set
  inline bool operator!() const { return !m_i; }

  // Test if specific bit(s) are set
  inline bool test( enum_t f ) const
  { return ( m_i & f ) == f && ( f != 0 || m_i == 0 ); }

protected:
  int_t m_i;
};

} // namespace vital
} // namespace kwiver

#endif
