// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "types.h"

/**
 * \file types.cxx
 *
 * \brief Implementation of base type logic.
 */

namespace sprokit {

pipeline_exception
::pipeline_exception() noexcept
: kwiver::vital::vital_exception()
{
}

pipeline_exception
::~pipeline_exception() noexcept
{
}

}
