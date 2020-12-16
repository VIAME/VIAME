// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining homography overlap helper functions
 */

#ifndef KWIVER_ARROWS_VXL_COMPUTE_HOMOGRAPHY_OVERLAP_H_
#define KWIVER_ARROWS_VXL_COMPUTE_HOMOGRAPHY_OVERLAP_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vnl/vnl_double_3x3.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// Return the overlap between two images.
/**
 * This function assumes that a homography perfectly describes the
 * transformation between these 2 images (in some reference coordinate
 * system). The overlap is returned as a percentage.
 */
KWIVER_ALGO_VXL_EXPORT
double
overlap( const vnl_double_3x3& h, const unsigned ni, const unsigned nj );

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
