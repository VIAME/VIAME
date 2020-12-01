// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
