// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to bounding box utilities
 */

#ifndef ARROWS_OCV_BOUNDING_BOX_H
#define ARROWS_OCV_BOUNDING_BOX_H

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/types/bounding_box.h>
#include <opencv2/core.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

/**
 * @brief Convert CvRect to bounding_box
 *
 * This operator converts a CvRect to a kwiver bounding box.
 *
 * @param vbox CvRect to convert
 *
 * @return Equivalent bounding box.
 */
template <typename T>
kwiver::vital::bounding_box<T> convert( const cv::Rect& vbox )
{
  typename kwiver::vital::bounding_box<T>::vector_type bb_tl( vbox.x, vbox.y );
  return kwiver::vital::bounding_box<T>( bb_tl, vbox.width, vbox.height );
}

// ------------------------------------------------------------------
/**
 * @brief Convert bounding box to CvRect
 *
 * @param bbox Bounding box to convert
 *
 * @return Equivalent CvRect
 */
template <typename T>
cv::Rect convert(const kwiver::vital::bounding_box<T>& bbox )
{
  // Note that CvRect has integer values. If T is a floating type. the
  // fractions are turncated.
  return { static_cast< int >( bbox.min_x() ),
           static_cast< int >( bbox.min_y() ),
           static_cast< int >( bbox.width() ),
           static_cast< int >( bbox.height() ) };
}

} } } // end namespace

#endif /* ARROWS_OCV_BOUNDING_BOX_H */
