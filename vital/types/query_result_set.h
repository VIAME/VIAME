// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the interface for a query result set.
 */

#ifndef VITAL_QUERY_RESULT_SET_H_
#define VITAL_QUERY_RESULT_SET_H_

#include "query_result.h"

namespace kwiver {
namespace vital {

/// Shared pointer to query result set
typedef std::vector< query_result_sptr > query_result_set;

/// Shared pointer to query result set
typedef std::shared_ptr< query_result_set > query_result_set_sptr;

} } // end namespace vital

#endif // VITAL_QUERY_RESULT_SET_H_
