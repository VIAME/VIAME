// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface vital::descriptor_set helpers
 */

#ifndef VITAL_C_HELPERS_DESCRIPTOR_SET_H_
#define VITAL_C_HELPERS_DESCRIPTOR_SET_H_

#include <vital/types/descriptor_set.h>
#include <vital/bindings/c/types/descriptor_set.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

extern SharedPointerCache< kwiver::vital::descriptor_set, vital_descriptor_set_t >
  DESCRIPTOR_SET_SPTR_CACHE;

}
}

#endif //VITAL_C_HELPERS_DESCRIPTOR_SET_H_
