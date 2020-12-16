// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the close_loops_multi_method algorithm
 */

#ifndef KWIVER_ARROWS_CORE_CLOSE_LOOPS_MULTI_METHOD_H_
#define KWIVER_ARROWS_CORE_CLOSE_LOOPS_MULTI_METHOD_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/track_set.h>

#include <vital/algo/match_features.h>
#include <vital/algo/close_loops.h>

#include <vital/config/config_block.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Attempts to stitch over incomplete or bad input frames.
/**
 * This class can run multiple other close_loops algorithm implementations
 * in attempt to accomplish this.
 */
class KWIVER_ALGO_CORE_EXPORT close_loops_multi_method
  : public vital::algo::close_loops
{
public:
  PLUGIN_INFO( "multi_method",
               "Iteratively run multiple loop closure algorithms." )

  /// Default Constructor
  close_loops_multi_method();

  /// Destructor
  virtual ~close_loops_multi_method() = default;

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  /**
   * This base virtual function implementation returns an empty configuration
   * block whose name is set to \c this->type_name.
   *
   * \returns \c config_block containing the configuration for this algorithm
   *          and any nested components.
   */
  virtual vital::config_block_sptr get_configuration() const;

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
  virtual void set_configuration(vital::config_block_sptr config);

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
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Run all internal loop closure algorithms.
  /**
   * \param frame_number the frame number of the current frame
   * \param image image data for the current frame
   * \param input the input feature track set to stitch
   * \param mask Optional mask image where positive values indicate
   *             regions to consider in the input image.
   * \returns an updated set of feature tracks after the stitching operation
   */
  virtual vital::feature_track_set_sptr
  stitch( vital::frame_id_t frame_number,
          vital::feature_track_set_sptr input,
          vital::image_container_sptr image,
          vital::image_container_sptr mask = vital::image_container_sptr() ) const;

private:

  /// Number of close loops methods we want to use.
  unsigned count_;

  /// The close loops methods to use.
  std::vector< vital::algo::close_loops_sptr > methods_;

};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
