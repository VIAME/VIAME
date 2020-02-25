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

#include <vital/types/point.h>

#include <iomanip>
#include <stdexcept>

namespace kwiver {
namespace vital {

template class point< 2, int >;
template class point< 2, double >;
template class point< 2, float >;
template class point< 3, double >;
template class point< 3, float >;
template class point< 4, double >;
template class point< 4, float >;

// ----------------------------------------------------------------------------
template < unsigned N, typename T >
std::ostream&
out(std::ostream& str, point< N, T >const& p)
{
  str << "point" << N << "D\n";

  auto const v = p.value();
  str << " - value : ";
  switch (N)
  {
  case 2:
    str << "[ " << v[0] << ", " << v[1] << " ]\n";
    break;
  case 3:
    str << "[ " << v[0] << ", " << v[1] << ", " << v[2] << " ]\n";
    break;
  case 4:
    str << "[ " << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << " ]\n";
    break;
  }

  str << " - covariance : ";
  auto const c = p.covariance().matrix();
  switch (N)
  {
  case 2:
    str << "[ " << c(0, 0) << ", " << c(0, 1) << "\n                  "
                << c(1, 0) << ", " << c(1, 1) << " ]\n";
    break;
  case 3:
    str << "[ " << c(0, 0) << ", " << c(0, 1) << ", " << c(0, 2) << "\n                  "
                << c(1, 0) << ", " << c(1, 1) << ", " << c(1, 2) << "\n                  "
                << c(2, 0) << ", " << c(2, 1) << ", " << c(2, 2) << " ]\n";
    break;
  case 4:
    str << "[ " << c(0, 0) << ", " << c(0, 1) << ", " << c(0, 2) << ", " << c(0, 3) << "\n                  "
                << c(1, 0) << ", " << c(1, 1) << ", " << c(1, 2) << ", " << c(1, 3) << "\n                  "
                << c(2, 0) << ", " << c(2, 1) << ", " << c(2, 2) << ", " << c(2, 3) << "\n                  "
                << c(3, 0) << ", " << c(3, 1) << ", " << c(3, 2) << ", " << c(3, 3) << " ]\n";
    break;
  }

  return str;
}

// ----------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& str, const point_2i& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_2d& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_2f& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_3d& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_3f& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_4d& p) { return out(str, p); }
std::ostream& operator<<(std::ostream& str, const point_4f& p) { return out(str, p); }

} } // end namespace
