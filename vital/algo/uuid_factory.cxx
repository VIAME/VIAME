// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for uid factory
 */

#include <vital/algo/uuid_factory.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

uuid_factory
::uuid_factory()
{
  attach_logger( "algo.uuid_factory" );
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::uuid_factory);
/// \endcond
