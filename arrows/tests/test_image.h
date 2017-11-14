/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
 * \brief test VXL image class functionality
 */

#include <vital/types/image.h>

#include <cmath>

namespace {

// ----------------------------------------------------------------------------
double value_at( int i, int j, int p )
{
  static constexpr double pi = 3.14159265358979323846;

  auto const w = 0.1 * static_cast<double>( p + 1 );
  auto const u = std::sin( pi * static_cast<double>( i ) * w );
  auto const v = std::sin( pi * static_cast<double>( j ) * w );
  return 0.5 * ( ( u * v ) + 1.0 );
}

// ----------------------------------------------------------------------------
// Helper function to populate the image with a pattern; the dynamic range is
// stretched between minv and maxv
template <typename T>
void
populate_vital_image(kwiver::vital::image& img, T minv, T maxv)
{
  const double range = static_cast<double>(maxv) - static_cast<double>(minv);
  const double offset = - static_cast<double>(minv);
  for( unsigned int p=0; p<img.depth(); ++p )
  {
    for( unsigned int j=0; j<img.height(); ++j )
    {
      for( unsigned int i=0; i<img.width(); ++i )
      {
        img.at<T>(i,j,p) = static_cast<T>(value_at(i, j, p) * range + offset);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// helper function to populate the image with a pattern
template <typename T>
void
populate_vital_image(kwiver::vital::image& img)
{
  const T minv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::min() : T(0);
  const T maxv = std::numeric_limits<T>::is_integer ? std::numeric_limits<T>::max() : T(1);
  populate_vital_image<T>(img, minv, maxv);
}

} // end anonymous namespace
