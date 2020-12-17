// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_type class
 */

#ifndef VITAL_DETECTED_OBJECT_TYPE_H_
#define VITAL_DETECTED_OBJECT_TYPE_H_

#include <vital/types/class_map.h>

#include <vital/vital_export.h>

namespace kwiver {

namespace vital {

struct detected_object_type_tag {};

extern template class VITAL_EXPORT class_map< detected_object_type_tag >;

using detected_object_type = class_map< detected_object_type_tag >;

// typedef for a class_map shared pointer
using detected_object_type_sptr = std::shared_ptr< detected_object_type >;
using detected_object_type_scptr = std::shared_ptr< detected_object_type const >;

} // namespace vital

} // namespace kwiver

#endif
