// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_CORE_FILTER_FEATURES_NONMAX_H_
#define KWIVER_ARROWS_CORE_FILTER_FEATURES_NONMAX_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/algo/filter_features.h>

/// \file
/// \brief Header for filtering features with non-max suppression

namespace kwiver {

namespace arrows {

namespace core {

/// \brief Algorithm that filters features using non-max suppression
class VIAME_IMAGE_PROCESSING_EXPORT filter_features_nonmax
  : public vital::algo::filter_features
{
public:
  PLUGGABLE_IMPL(
    filter_features_nonmax,
    "Filter features using non-max supression.",
    PARAM_DEFAULT(
      suppression_radius, double,
      "The radius, in pixels, within which to "
      "suppress weaker features.  This is an initial guess. "
      "The radius is adapted to reach the desired number of "
      "features.  If target_num_features is 0 then this radius "
      "is not adapted.",
      0.0 ),
    PARAM_DEFAULT(
      num_features_target, size_t,
      "The target number of features to detect. "
      "The suppression radius is dynamically adjusted to "
      "acheive this number of features.",
      500 ),
    PARAM_DEFAULT(
      num_features_range, size_t,
      "The number of features above target_num_features to "
      "allow in the output.  This window allows the binary "
      "search on radius to terminate sooner.",
      50 ),
    PARAM_DEFAULT(
      resolution, size_t,
      "The resolution (N) of the filter for computing neighbors."
      " The filter is an (2N+1) x (2N+1) box containing a circle"
      " of radius N. The value must be a positive integer. "
      "Larger values are more "
      "accurate at the cost of more memory and compute time.",
      3 ),
  )

  /// Destructor
  virtual ~filter_features_nonmax();

  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

protected:
  /// filter a feature set
  ///
  /// \param [in] feature set to filter
  /// \param [out] indices of the kept features to the original feature set
  /// \returns a filtered version of the feature set
  virtual vital::feature_set_sptr
  filter(
    vital::feature_set_sptr input,
    std::vector< size_t >& indices ) const;
  using filter_features::filter;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace core

} // end namespace arrows

} // end namespace kwiver

#endif // KWIVER_ARROWS_CORE_FILTER_FEATURES_NONMAX_H_
