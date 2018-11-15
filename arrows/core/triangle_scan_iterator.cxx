/*ckwg +29
 * Copyright 2018 by Kitware, SAS.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief Implementation of core triangle_scan_iterator
 */

#include "triangle_scan_iterator.h"

namespace {

void min_max(double a, double b, double c, double &min, double &max)
{
  min = std::min(std::min(a, b), c);
  max = std::max(std::max(a, b), c);
}
}

namespace kwiver {
namespace arrows {
namespace core {

void triangle_scan_iterator::reset()
{
  double min, max;
  min_max(a(0), b(0), c(0), min, max);
  x0 = static_cast<int>(std::ceil(min));
  x1 = static_cast<int>(std::floor(max));

  min_max(a(1), b(1), c(1), min, max);
  y0 = static_cast<int>(std::ceil(min));
  y1 = static_cast<int>(std::floor(max));

  scan_y_ = y0 - 1;

  g = ((a + b + c) / 3).array().floor();

  vital::vector_2d ga(a(0) - g(0), a(1) - g(1));
  vital::vector_2d gb(b(0) - g(0), b(1) - g(1));
  vital::vector_2d gc(c(0) - g(0), c(1) - g(1));

  data[0][0] = gb(1) - gc(1); data[0][1] = gc(0) - gb(0); data[0][2] = gb(0) * gc(1) - gb(1) * gc(0);
  data[1][0] = gc(1) - ga(1); data[1][1] = ga(0) - gc(0); data[1][2] = gc(0) * ga(1) - gc(1) * ga(0);
  data[2][0] = ga(1) - gb(1); data[2][1] = gb(0) - ga(0); data[2][2] = ga(0) * gb(1) - ga(1) * gb(0);
  double tmp = (gb(0) * gc(1) - gb(0) * ga(1) - ga(0) * gc(1)
                - gc(0) * gb(1) + gc(0) * ga(1) + ga(0) * gb(1));

  if (tmp < 0)
    tmp = -1;
  else
    tmp = 1;

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      data[i][j] *= tmp;
    }
  }
}

bool triangle_scan_iterator::next()
{
  if (++scan_y_ > y1)
    return false;

  double xmin = x0 - g(0);
  double xmax = x1 - g(0);
  for (auto & i : data)
  {
    double a_ = i[0];
    double b_ = i[1] * (scan_y_ - g(1)) + i[2];
    if (std::abs(a_) > 1e-5)
    {
      double x = -b_ / a_;
      if (a_ > 0)
      {
        if (x > xmin)
          xmin = x;
      }
      else
      {
        if (x < xmax)
          xmax = x;
      }
    }
  }
  start_x_ = static_cast<int>(std::ceil(xmin + g(0)));
  end_x_ = static_cast<int>(std::floor(xmax + g(0)));
  return true;
}

}
}
}
