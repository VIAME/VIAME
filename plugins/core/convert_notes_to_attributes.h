/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Helper functions for dealing with notes and attributes in viame csvs
 */

#ifndef VIAME_CORE_CONVERT_NOTES_TO_ATTRIBUTES_H
#define VIAME_CORE_CONVERT_NOTES_TO_ATTRIBUTES_H

#include <plugins/core/viame_core_export.h>

#include <vital/types/detected_object.h>

#include <string>
#include <memory>
#include <vector>

namespace viame
{


VIAME_CORE_EXPORT std::string
notes_to_attributes( const std::vector< std::string >& notes,
                     const std::string delim = "," );


VIAME_CORE_EXPORT void
add_attributes_to_detection( kwiver::vital::detected_object& detection,
                             const std::vector< std::string >& attrs );


} // end namespace

#endif // VIAME_CORE_CONVERT_NOTES_TO_ATTRIBUTES_H
