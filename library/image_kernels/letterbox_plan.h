/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is"
and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any
direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
// Lanczos coefficients adapted from OpenCV 4.9 modules/imgproc/src/resize.cpp.
// Host planning keeps CPU/GPU coefficient rounding identical.
#ifndef VIAME_LETTERBOX_PLAN_H
#define VIAME_LETTERBOX_PLAN_H
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <vector>
namespace viame {
namespace image_kernels {
namespace detail {
struct resize_entry { int index; float weight; int fixed; };
static inline void interpolateLanczos4(float x, float *coeffs) {
  static const double s45 = 0.70710678118654752440084436210485;
  static const double cs[][2] = {{1, 0},  {-s45, -s45}, {0, 1},  {s45, -s45},
                                 {-1, 0}, {s45, s45},   {0, -1}, {-s45, s45}};

  float sum = 0;
  double y0 = -(x + 3) * 3.1415926535897932384626433832795 * 0.25,
         s0 = std::sin(y0), c0 = std::cos(y0);
  for (int i = 0; i < 8; i++) {
    float y0_ = (x + 3 - i);
    if (fabs(y0_) >= 1e-6f) {
      double y = -y0_ * 3.1415926535897932384626433832795 * 0.25;
      coeffs[i] = (float)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
    } else {
      // special handling for 'x' values:
      // - ~0.0: 0 0 0 1 0 0 0 0
      // - ~1.0: 0 0 0 0 1 0 0 0
      coeffs[i] = 1e30f;
    }
    sum += coeffs[i];
  }

  sum = 1.f / sum;
  for (int i = 0; i < 8; i++)
    coeffs[i] *= sum;
}

struct resize_axis {
  std::vector<int> offsets;
  std::vector<resize_entry> entries;
};
inline resize_axis make_resize_axis(int source, int dest, bool area) {
  resize_axis result;
  double scale = 1.0 / (double(dest) / source);
  for (int d = 0; d < dest; ++d) {
    result.offsets.push_back(static_cast<int>(result.entries.size()));
    if (area) {
      double start = d * scale, end = start + scale,
             cell = std::min(scale, source - start);
      int first = static_cast<int>(std::ceil(start)),
          last = static_cast<int>(std::floor(end));
      last = std::min(last, source - 1);
      first = std::min(first, last);
      if (first - start > 1e-3)
        result.entries.push_back({first - 1, float((first - start) / cell), 0});
      for (int s = first; s < last; ++s)
        result.entries.push_back({s, float(1.0 / cell), 0});
      if (end - last > 1e-3)
        result.entries.push_back(
            {last, float(std::min({end - last, 1.0, cell}) / cell), 0});
    } else {
      float fraction = float((d + 0.5) * scale - 0.5);
      int first = static_cast<int>(std::floor(fraction));
      fraction -= first;
      float taps[8];
      interpolateLanczos4(fraction, taps);
      for (int k = 0; k < 8; ++k)
        result.entries.push_back({std::clamp(first + k - 3, 0, source - 1), taps[k],
                                  int(std::lrint(taps[k] * 2048.f))});
    }
  }
  result.offsets.push_back(static_cast<int>(result.entries.size()));
  return result;
}
struct letterbox_plan {
  int width, height, left, top;
  bool area;
  resize_axis x, y;
};
inline letterbox_plan make_letterbox_plan(int sw, int sh, int dw, int dh) {
  double scale = std::min(double(dw) / sw, double(dh) / sh);
  letterbox_plan p;
  p.width = int(std::nearbyint(sw * scale));
  p.height = int(std::nearbyint(sh * scale));
  if (p.width < 1 || p.height < 1)
    throw std::invalid_argument("letterbox embedded image has no area");
  p.left = int(std::nearbyint((dw - p.width) / 2.0));
  p.top = int(std::nearbyint((dh - p.height) / 2.0));
  p.area = scale < 1;
  p.x = make_resize_axis(sw, p.width, p.area);
  p.y = make_resize_axis(sh, p.height, p.area);
  return p;
}
} // namespace detail
} // namespace image_kernels
} // namespace viame
#endif
