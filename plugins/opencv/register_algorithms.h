/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_REGISTER_ALGORITHMS_H
#define VIAME_OPENCV_REGISTER_ALGORITHMS_H

#include "viame_opencv_export.h"

#include <vital/registrar.h>

namespace viame {

// Register core algorithms with the given or global registrar
VIAME_OPENCV_EXPORT
int register_algorithms( kwiver::vital::registrar &reg
  = kwiver::vital::registrar::instance() );

} // end namespace

#endif /* VIAME_OPENCV_REGISTER_ALGORITHMS_H */
