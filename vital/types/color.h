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

/// Struct to represent an RGB tuple
struct rgb_color
{
  /// Default constructor - set the color to white
  rgb_color() : r(255), g(255), b(255) {}

  /// Constructor
  rgb_color(uint8_t const &cr,
            uint8_t const &cg,
            uint8_t const &cb)
    : r(cr), g(cg), b(cb) {}

  /// Copy Constructor
  rgb_color( rgb_color const &c )
    : r(c.r), g(c.g), b(c.b) {}

  /// Serialization of the class data
  template<class Archive>
  void serialize(Archive & archive)
  {
    archive( r, g, b );
  }

  uint8_t r;
  uint8_t g;
  uint8_t b;
};

/// comparison operator for an rgb_color
inline
bool
operator==( rgb_color const& c1, rgb_color const& c2 )
{
  return (c1.r == c2.r) && (c1.g == c2.g) && (c1.b == c2.b);
}

/// comparison operator for an rgb_color
inline
bool
operator!=( rgb_color const& c1, rgb_color const& c2 )
{
  return !(c1 == c2);
}

/// output stream operator for an rgb_color
inline
std::ostream&
operator<<( std::ostream& s, const rgb_color& c )
{
  // Note the '+' prefix here is used to print characters
  // as decimal number, not ASCII characters
  s << +c.r << " " << +c.g << " " << +c.b;
  return s;
}

/// input stream operator for an rgb_color
inline
std::istream&
operator>>( std::istream& s, rgb_color& c )
{
  int rv = 255, gv = 255, bv = 255;
  s >> rv >> gv >> bv;
  c.r = static_cast<uint8_t>( rv );
  c.g = static_cast<uint8_t>( gv );
  c.b = static_cast<uint8_t>( bv );
  return s;
}

} } // end namespace vital

#endif // VITAL_COLOR_H_
