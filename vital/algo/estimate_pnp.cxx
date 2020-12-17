// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief estimate_pnp instantiation
 */

#include <vital/algo/estimate_pnp.h>
#include <vital/algo/algorithm.txx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::estimate_pnp);
/// \endcond

namespace kwiver {
namespace vital {
namespace algo {

estimate_pnp
::estimate_pnp()
{
  attach_logger( "algo.estimate_pnp" );
}

} } } // end namespace
