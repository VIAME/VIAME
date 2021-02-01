// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_CORE_FILTER_FEATURES_NONMAX_H_
#define KWIVER_ARROWS_CORE_FILTER_FEATURES_NONMAX_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/filter_features.h>

/**
 * \file
 * \brief Header for filtering features with non-max suppression
 */

namespace kwiver {
namespace arrows {
namespace core {

/// \brief Algorithm that filters features using non-max suppression
class KWIVER_ALGO_CORE_EXPORT filter_features_nonmax
  : public vital::algo::filter_features
{
public:
  PLUGIN_INFO( "nonmax",
               "Filter features using non-max supression." )

  /// Constructor
  filter_features_nonmax();

  /// Destructor
  virtual ~filter_features_nonmax();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

protected:

  /// filter a feature set
  /**
   * \param [in] feature set to filter
   * \param [out] indices of the kept features to the original feature set
   * \returns a filtered version of the feature set
   */
  virtual vital::feature_set_sptr
  filter(vital::feature_set_sptr input, std::vector<unsigned int> &indices) const;
  using filter_features::filter;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_CORE_FILTER_FEATURES_NONMAX_H_
