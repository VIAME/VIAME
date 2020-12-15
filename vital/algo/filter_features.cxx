// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Instantiation of \link kwiver::vital::algo::algorithm_def algorithm_def<T>
 *        \endlink for \link kwiver::vital::algo::filter_features
 *        filter_features \endlink
 */

#include <vital/algo/filter_features.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

filter_features
::filter_features()
{
  attach_logger( "algo.filter_features" );
}

feature_set_sptr
filter_features
::filter(feature_set_sptr feat) const
{
  std::vector<unsigned int> indices;
  return filter(feat, indices);
}

std::pair<feature_set_sptr, descriptor_set_sptr>
filter_features
::filter( feature_set_sptr feat, descriptor_set_sptr descr) const
{
  std::vector<unsigned int> indices;
  feature_set_sptr filt_feat = filter(feat, indices);

  // Iterate through descriptor sptrs, keeping those in the same index as the
  // kept features.
  std::vector<descriptor_sptr> filtered_descr;
  filtered_descr.reserve(indices.size());

  for (unsigned int i = 0; i < indices.size(); i++)
  {
    filtered_descr.push_back( descr->at( indices[i] ) );
  }

  descriptor_set_sptr filt_descr = std::make_shared<kwiver::vital::simple_descriptor_set>(
                                     kwiver::vital::simple_descriptor_set(filtered_descr));

  return std::make_pair(filt_feat, filt_descr);
}

} } } // end namespace

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::filter_features);
