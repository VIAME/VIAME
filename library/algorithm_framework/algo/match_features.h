// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief match_features algorithm definition interface

#ifndef VITAL_ALGO_MATCH_FEATURES_H_
#define VITAL_ALGO_MATCH_FEATURES_H_

#include <viame/algorithm_framework/vital_config.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/feature_set.h>
#include <viame/core_types/match_set.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for matching feature points
class VITAL_ALGO_EXPORT match_features
  : public kwiver::vital::algorithm
{
public:
  match_features();
  PLUGGABLE_INTERFACE( match_features );
  /// Match one set of features and corresponding descriptors to another
  ///
  /// \param feat1 the first set of features to match
  /// \param desc1 the descriptors corresponding to \a feat1
  /// \param feat2 the second set fof features to match
  /// \param desc2 the descriptors corresponding to \a feat2
  /// \returns a set of matching indices from \a feat1 to \a feat2
  virtual kwiver::vital::match_set_sptr
  match(
    kwiver::vital::feature_set_sptr feat1,
    kwiver::vital::descriptor_set_sptr desc1,
    kwiver::vital::feature_set_sptr feat2,
    kwiver::vital::descriptor_set_sptr desc2 ) const = 0;
};

/// Shared pointer type for match_features algorithm definition class
typedef std::shared_ptr< match_features > match_features_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_MATCH_FEATURES_H_
