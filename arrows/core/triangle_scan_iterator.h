// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for kwiver::arrows::triangle_scan_iterator
 */

#ifndef KWIVER_ARROWS_CORE_TRIANGLE_SCAN_ITERATOR_H
#define KWIVER_ARROWS_CORE_TRIANGLE_SCAN_ITERATOR_H

#include <arrows/core/kwiver_algo_core_export.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Provides access to the pixels of a triangle using scanlines
class KWIVER_ALGO_CORE_EXPORT triangle_scan_iterator
{
public:
  triangle_scan_iterator(vital::vector_2d const & pt1,
                         vital::vector_2d const & pt2,
                         vital::vector_2d const & pt3) :
    a(pt1), b(pt2), c(pt3)
  {
    reset();
  }

  /// Reset the iterator state
  void reset();

  /// Update the iterator to the next scanline
  bool next();

  /// Current scanline index
  int scan_y() const { return scan_y_; }

  /// Index of the first pixel of the current scanline
  int start_x() const { return start_x_; }

  /// Index of the last pixel of the current scanline
  int end_x() const { return end_x_; }

private:
  vital::vector_2d const &a, &b, &c;
  vital::vector_2d g;
  int scan_y_;
  int start_x_, end_x_;
  int x0, y0, x1, y1;
  double data[3][3];
};

}
}
}

#endif // KWIVER_ARROWS_CORE_TRIANGLE_SCAN_ITERATOR_H
