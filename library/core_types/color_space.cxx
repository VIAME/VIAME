/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 * used
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

#include "viame/core_types/color_space.h"

#include <algorithm>
#include <map>

/// \brief Converts a string to a known color space if possible.
viame::color_space
viame
::string_to_color_space( const std::string& str )
{
  std::string lowercase_str = str;

  std::transform(
    lowercase_str.begin(),
    lowercase_str.end(),
    lowercase_str.begin(),
    ::tolower );

  // Manually create a short branch tree. Another common way to create
  // a string to enum correspondance would be to use a static std::map,
  // but that would incur a tiny amount of extra run-time memory.
  if( lowercase_str.size() < 3 )
  {
    return viame::INVALID_CS;
  }

  if( lowercase_str[ 0 ] < 'k' )
  {
    if( lowercase_str == "bgr" )
    {
      return viame::BGR;
    }
    if( lowercase_str == "cmyk" )
    {
      return viame::CMYK;
    }
    if( lowercase_str == "hls" )
    {
      return viame::HLS;
    }
    if( lowercase_str == "hsl" )
    {
      return viame::HSL;
    }
    if( lowercase_str == "hsv" )
    {
      return viame::HSV;
    }
  }
  else
  {
    if( lowercase_str == "lab" )
    {
      return viame::Lab;
    }
    if( lowercase_str == "luv" )
    {
      return viame::Luv;
    }
    if( lowercase_str == "rgb" )
    {
      return viame::RGB;
    }
    if( lowercase_str == "xyz" )
    {
      return viame::XYZ;
    }
    if( lowercase_str == "ycrcb" )
    {
      return viame::YCrCb;
    }
    if( lowercase_str == "ycbcr" )
    {
      return viame::YCbCr;
    }
  }

  return viame::INVALID_CS;
}

/// \brief Converts a known color space to a string
std::string
viame
::color_space_to_string( const viame::color_space cs )
{
  const static std::map< viame::color_space,
    std::string > mapping = { { viame::INVALID_CS, "INVALID" },
    { viame::BGR, "BGR" },
    { viame::CMYK, "CMYK" },
    { viame::HLS, "HLS" },
    { viame::HSL, "HSL" },
    { viame::HSV, "HSV" },
    { viame::Lab, "Lab" },
    { viame::Luv, "Luv" },
    { viame::RGB, "RGB" },
    { viame::XYZ, "XYZ" },
    { viame::YCrCb, "YCrCb" },
    { viame::YCbCr, "YCbCr" } };

  return mapping.at( cs );
}
