/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_CONVERT_POLYGONS_TO_MASK_H
#define VIAME_CORE_CONVERT_POLYGONS_TO_MASK_H

#include "viame_core_export.h"

#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <vital/types/bounding_box.h>

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
VIAME_CORE_EXPORT
void convert_polys_to_mask( const std::vector< std::string >& polygons,
                            const kwiver::vital::bounding_box_d& bbox,
                            kwiver::vital::image_of< uint8_t >& output );


} // end namespace viame

#endif // VIAME_CORE_CONVERT_POLYGONS_TO_MASK_H
