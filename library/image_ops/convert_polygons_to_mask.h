/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_CONVERT_POLYGONS_TO_MASK_H
#define VIAME_IMAGE_OPS_CONVERT_POLYGONS_TO_MASK_H

#include "viame_image_ops_export.h"

#include <viame/core_types/image.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/bounding_box.h>

#include <string>
#include <vector>

namespace viame
{

/// Convert a set of polygons in string form into an output mask
///
/// @param polygons Input polygons
/// @param bbox Box region for mask
/// @param output Output mask image data
///
/// @throws runtime_error on invalid or unable to parse filename format
///
VIAME_IMAGE_OPS_EXPORT
void convert_polys_to_mask( const std::vector< std::string >& polygons,
                            const kwiver::vital::bounding_box_d& bbox,
                            kwiver::vital::image_of< uint8_t >& output );


} // end namespace viame

#endif // VIAME_IMAGE_OPS_CONVERT_POLYGONS_TO_MASK_H
