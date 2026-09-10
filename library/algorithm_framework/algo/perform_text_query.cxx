// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief perform_text_query algorithm instantiation

#include <viame/algorithm_framework/algo/perform_text_query.h>

namespace kwiver {

namespace vital {

namespace algo {

perform_text_query
::perform_text_query()
{
  attach_logger( "algo.perform_text_query" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
