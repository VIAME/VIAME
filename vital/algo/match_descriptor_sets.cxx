// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of bag of words matching
 */

#include <vital/algo/match_descriptor_sets.h>
#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace vital {
namespace algo {

match_descriptor_sets
::match_descriptor_sets()
{
  attach_logger( "algo.match_descriptor_sets" );
}

std::vector<vital::frame_id_t>
match_descriptor_sets
::query_and_append(const vital::descriptor_set_sptr desc,
  frame_id_t frame)
{
  auto putative_matching_frames = this->query(desc);
  this->append_to_index(desc, frame);
  return putative_matching_frames;
}

} } } // end namespace

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::match_descriptor_sets);
/// \endcond
