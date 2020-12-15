// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File description here.
 */

#ifndef VITAL_C_HELPERS_FEATURE_H_
#define VITAL_C_HELPERS_FEATURE_H_

#include <vital/types/feature.h>
#include <vital/bindings/c/types/feature.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

extern SharedPointerCache< vital::feature, vital_feature_t >
  FEATURE_SPTR_CACHE;

}
}

#endif //VITAL_C_HELPERS_FEATURE_H_
