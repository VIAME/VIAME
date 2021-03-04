// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface vital::descriptor helpers
 */

#ifndef VITAL_C_HELPERS_DESCRIPTOR_H_
#define VITAL_C_HELPERS_DESCRIPTOR_H_

#include <vital/types/descriptor.h>
#include <vital/bindings/c/types/descriptor.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

extern SharedPointerCache< kwiver::vital::descriptor, vital_descriptor_t >
  DESCRIPTOR_SPTR_CACHE;

}
}

#endif //VITAL_C_HELPERS_DESCRIPTOR_H_
