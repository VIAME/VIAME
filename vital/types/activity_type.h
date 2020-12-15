// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for activity_type class
 */

#ifndef VITAL_ACTIVITY_TYPE_H_
#define VITAL_ACTIVITY_TYPE_H_

#include <vital/types/class_map.h>

#include <vital/vital_export.h>

namespace kwiver {

namespace vital {

struct activity_type_tag {};

extern template class VITAL_EXPORT class_map< activity_type_tag >;

using activity_type = class_map< activity_type_tag >;

// typedef for a class_map shared pointer
using activity_type_sptr = std::shared_ptr< activity_type >;
using activity_type_scptr = std::shared_ptr< activity_type const >;

} // namespace vital

} // namespace kwiver

#endif
