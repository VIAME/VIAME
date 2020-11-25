// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to bounding box utilities
 */

#ifndef ARROWS_VXL_BOUNDING_BOX_H
#define ARROWS_VXL_BOUNDING_BOX_H

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vgl/vgl_box_2d.h>
#include <vital/types/bounding_box.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/**
 * @brief Convert vgl_box_2d to bounding_box
 *
 * This operator converts a vgl_box_2d to a kwiver bounding box.
 *
 * @param vbox vgl_box_2d to convert
 *
 * @return Equivalent bounding box.
 */
template <typename T>
kwiver::vital::bounding_box<T> convert( const vgl_box_2d<T>& vbox )
{
  return kwiver::vital::bounding_box<T>( vbox.min_x(),
                                         vbox.min_y(),
                                         vbox.max_x(),
                                         vbox.max_y() );
}

// ------------------------------------------------------------------
/**
 * @brief Convert bounding box to vgl_box_2d
 *
 * @param bbox Bounding box to convert
 *
 * @return Equivalent vgl_box_2d
 */
template <typename T>
vgl_box_2d<T> convert(const kwiver::vital::bounding_box<T>& bbox )
{
  return vgl_box_2d<T>( bbox.min_x(), bbox.max_x(),
                        bbox.min_y(), bbox.max_y() );
}

} } } // end namespace

#endif /* ARROWS_VXL_BOUNDING_BOX_H */
