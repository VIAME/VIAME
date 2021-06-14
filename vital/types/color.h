// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Structs for representing color
 */

#ifndef VITAL_COLOR_H_
#define VITAL_COLOR_H_

#include <iostream>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Struct to represent an RGB tuple.
struct rgb_color
{
  /// Construct with a default value (white).
  rgb_color() = default;

  /// Construct with specified component values.
  rgb_color( uint8_t cr, uint8_t cg, uint8_t cb )
    : r{ cr }, g{ cg }, b{ cb } {}

  rgb_color( rgb_color const& ) = default;

  rgb_color& operator=( rgb_color const& ) = default;

  template < class Archive >
  void
  serialize( Archive& archive )
  {
    archive( r, g, b );
  }

  uint8_t r = 255;
  uint8_t g = 255;
  uint8_t b = 255;
};

// ----------------------------------------------------------------------------
/// Comparison operator for an rgb_color.
inline
bool
operator==( rgb_color const& c1, rgb_color const& c2 )
{
  return ( c1.r == c2.r ) && ( c1.g == c2.g ) && ( c1.b == c2.b );
}

// ----------------------------------------------------------------------------
/// Comparison operator for an rgb_color.
inline
bool
operator!=( rgb_color const& c1, rgb_color const& c2 )
{
  return !( c1 == c2 );
}

// ----------------------------------------------------------------------------
/// Output stream operator for an rgb_color.
inline
std::ostream&
operator<<( std::ostream& s, const rgb_color& c )
{
  // Note the '+' prefix here is used to promote the members to int so they are
  // printed as decimal numbers, rather than as characters (char)
  s << +c.r << ' ' << +c.g << ' ' << +c.b;
  return s;
}

// ----------------------------------------------------------------------------
/// Input stream operator for an rgb_color.
inline
std::istream&
operator>>( std::istream& s, rgb_color& c )
{
  int rv = 255, gv = 255, bv = 255;
  s >> rv >> gv >> bv;
  c.r = static_cast< uint8_t >( rv );
  c.g = static_cast< uint8_t >( gv );
  c.b = static_cast< uint8_t >( bv );
  return s;
}

} // namespace vital

} // namespace kwiver

#endif
