// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the core match_features_fundamental_matrix algorithm
 */

#ifndef KWIVER_ARROWS__MATCH_FEATURES_FUNDMENTAL_MATRIX_H_
#define KWIVER_ARROWS__MATCH_FEATURES_FUNDMENTAL_MATRIX_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/filter_features.h>

#include <vital/algo/estimate_fundamental_matrix.h>
#include <vital/algo/match_features.h>
#include <vital/config/config_block.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Combines a feature matcher, fundamental matrix estimation, and filtering
/**
 *  This is a meta-algorithm for feature matching that combines one other feature
 *  matcher with fundamental matrix estimation and feature filtering.
 *  The algorithm applies another configurable feature matcher algorithm and
 *  then applies a fundamental matrix estimation algorithm to the resulting matches.
 *  Outliers to the fit fundamental matrix are discarded from the set of matches.
 *
 *  If a filter_features algorithm is provided, this will be run on the
 *  input features \b before running the matcher.
 */
class KWIVER_ALGO_CORE_EXPORT match_features_fundamental_matrix
  : public vital::algo::match_features
{
public:
  PLUGIN_INFO( "fundamental_matrix_guided",
               "Use an estimated fundamental matrix as a geometric filter"
               " to remove outlier matches." )

  /// Default Constructor
  match_features_fundamental_matrix();

  /// Destructor
  virtual ~match_features_fundamental_matrix();

  /// Get this alg's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algo's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Match one set of features and corresponding descriptors to another
  /**
   * \param [in] feat1 the first set of features to match
   * \param [in] desc1 the descriptors corresponding to \a feat1
   * \param [in] feat2 the second set of features to match
   * \param [in] desc2 the descriptors corresponding to \a feat2
   * \returns a set of matching indices from \a feat1 to \a feat2
   */
  virtual vital::match_set_sptr
  match(vital::feature_set_sptr feat1, vital::descriptor_set_sptr desc1,
        vital::feature_set_sptr feat2, vital::descriptor_set_sptr desc2) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace algo
} // end namespace arrows
} // end namespace kwiver

#endif
