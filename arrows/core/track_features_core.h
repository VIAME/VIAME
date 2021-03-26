// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the track_features_core algorithm
 */

#ifndef ARROWS_PLUGINS_CORE_TRACK_FEATURES_CORE_H_
#define ARROWS_PLUGINS_CORE_TRACK_FEATURES_CORE_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/track_features.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A basic feature tracker
class KWIVER_ALGO_CORE_EXPORT track_features_core
  : public vital::algo::track_features
{
public:
  PLUGIN_INFO( "core",
               "Track features from frame to frame"
               " using feature detection, matching, and loop closure." )

  /// Default Constructor
  track_features_core();

  /// Destructor
  virtual ~track_features_core() noexcept;

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  /**
   * This base virtual function implementation returns an empty configuration
   * block whose name is set to \c this->type_name.
   *
   * \returns \c config_block containing the configuration for this algorithm
   *          and any nested components.
   */
  vital::config_block_sptr get_configuration() const override;

  /// Set this algorithm's properties via a config block
  /**
   * \throws no_such_configuration_value_exception
   *    Thrown if an expected configuration value is not present.
   * \throws algorithm_configuration_exception
   *    Thrown when the algorithm is given an invalid \c config_block or is'
   *    otherwise unable to configure itself.
   *
   * \param config  The \c config_block instance containing the configuration
   *                parameters for this algorithm
   */
  void set_configuration(vital::config_block_sptr config) override;

  /// Check that the algorithm's currently configuration is valid
  /**
   * This checks solely within the provided \c config_block and not against
   * the current state of the instance. This isn't static for inheritence
   * reasons.
   *
   * \param config  The config block to check configuration of.
   *
   * \returns true if the configuration check passed and false if it didn't.
   */
  bool check_configuration(vital::config_block_sptr config) const override;

  /// Extend a previous set of feature tracks using the current frame
  /**
   * \throws image_size_mismatch_exception
   *    When the given non-zero mask image does not match the size of the
   *    dimensions of the given image data.
   *
   * \param [in] prev_tracks the feature tracks from previous tracking steps
   * \param [in] frame_number the frame number of the current frame
   * \param [in] image_data the image pixels for the current frame
   * \param [in] mask Optional mask image that uses positive values to denote
   *                  regions of the input image to consider for feature
   *                  tracking. An empty sptr indicates no mask (default
   *                  value).
   * \returns an updated set of feature tracks including the current frame
   */
  vital::feature_track_set_sptr
  track(vital::feature_track_set_sptr prev_tracks,
        vital::frame_id_t frame_number,
        vital::image_container_sptr image_data,
        vital::image_container_sptr mask = {}) const override;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
