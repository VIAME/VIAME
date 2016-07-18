/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
  rgb_color(unsigned char const &cr,
            unsigned char const &cg,
            unsigned char const &cb)
    : r(cr), g(cg), b(cb) {}

  /// Copy Constructor
  rgb_color( rgb_color const &c )
    : r(c.r), g(c.g), b(c.b) {}

  unsigned char r;
  unsigned char g;
  unsigned char b;
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
  c.r = static_cast<unsigned char>( rv );
  c.g = static_cast<unsigned char>( gv );
  c.b = static_cast<unsigned char>( bv );
  return s;
}




} } // end namespace vital

#endif // VITAL_COLOR_H_
