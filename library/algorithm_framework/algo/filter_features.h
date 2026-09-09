// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_FILTER_FEATURES_H_
#define VITAL_ALGO_FILTER_FEATURES_H_

#include <viame/algorithm_framework/vital_config.h>

#include <memory>
#include <utility>
#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/feature_set.h>

/// \file
/// \brief Header defining abstract \link kwiver::vital::algo::filter_features
///        filter features \endlink algorithm

namespace kwiver {

namespace vital {

namespace algo {

/// \brief Abstract base class for feature set filter algorithms.
class VITAL_ALGO_EXPORT filter_features
  : public kwiver::vital::algorithm
{
public:
  /// Return the name of this algorithm.
  filter_features();
  PLUGGABLE_INTERFACE( filter_features );
  /// Filter a feature set and return a subset of the features
  ///
  /// The default implementation call the pure virtual function
  /// filter(feature_set_sptr feat, std::vector<size_t> &indices) const
  /// \param [in] input The feature set to filter
  /// \returns a filtered version of the feature set (simple_feature_set)
  virtual kwiver::vital::feature_set_sptr
  filter( kwiver::vital::feature_set_sptr input ) const;

  /// Filter a feature_set and its coresponding descriptor_set
  ///
  /// The default implementation calls
  /// filter(feature_set_sptr feat, std::vector<size_t> &indices) const
  /// using with \p feat and then uses the resulting \p indices to construct
  /// a simple_descriptor_set with the corresponding descriptors.
  /// \param [in] feat The feature set to filter
  /// \param [in] descr The parallel descriptor set to filter
  /// \returns a pair of the filtered features and descriptors
  using filter_return_value = std::pair< kwiver::vital::feature_set_sptr,
    kwiver::vital::descriptor_set_sptr >;
  virtual filter_return_value
  filter(
    kwiver::vital::feature_set_sptr feat,
    kwiver::vital::descriptor_set_sptr descr ) const;

protected:
  /// Filter a feature set and return a new feature set with a subset of
  /// features
  ///
  /// \param [in] feat The input feature set
  /// \param [in,out] indices The indices into \p feat of the features retained
  /// \return a new feature set containing the subset of features noted by \p
  /// indices
  virtual kwiver::vital::feature_set_sptr
  filter(
    kwiver::vital::feature_set_sptr feat,
    std::vector< size_t >& indices ) const = 0;
};

/// type definition for shared pointer to a filter_features algorithm
typedef std::shared_ptr< filter_features > filter_features_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_FILTER_FEATURES_H_
