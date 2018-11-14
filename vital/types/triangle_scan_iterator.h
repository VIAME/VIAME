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
 * \brief Abstract triangle_scan_iterator interface
 */

#ifndef VITAL_TYPES_TRIANGLE_SCAN_ITERATOR_H
#define VITAL_TYPES_TRIANGLE_SCAN_ITERATOR_H

#include <vital/vital_export.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace vital {

class VITAL_EXPORT triangle_scan_iterator
{
public:
  triangle_scan_iterator(const vector_2d& pt1, const vector_2d& pt2, const vector_2d& pt3) :
    a(pt1), b(pt2), c(pt3)
  {
    reset();
  }

  void reset();
  bool next();
  int scan_y() const { return scan_y_; }
  int start_x() const { return start_x_; }
  int end_x() const { return end_x_; }

private:
  const vector_2d &a, &b, &c;
  vector_2d g;
  int scan_y_;
  int start_x_, end_x_;
  int x0, y0, x1, y1;
  double data[3][3];
};




}
}


#endif // VITAL_TYPES_TRIANGLE_SCAN_ITERATOR_H
