// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the compute_ref_homography algorithm
 */

#ifndef KWIVER_ARROWS_CORE_COMPUTE_REF_HOMOGRAPHY_CORE_H_
#define KWIVER_ARROWS_CORE_COMPUTE_REF_HOMOGRAPHY_CORE_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/algorithm.h>
#include <vital/algo/compute_ref_homography.h>
#include <vital/types/homography.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Default impl class for mapping each image to some reference image.
/**
 * This class differs from estimate_homographies in that estimate_homographies
 * simply performs a homography regression from matching feature points. This
 * class is designed to generate different types of homographies from input
 * feature tracks, which can transform each image back to the same coordinate
 * space derived from some initial refrerence image.
 *
 * This implementation is state-based and is meant to be run in an online
 * fashion, i.e. run against a track set that has been iteratively updated on
 * successive non-regressing frames. This is ideal for when it is desired to
 * compute reference frames on all frames in a sequence.
 */
class KWIVER_ALGO_CORE_EXPORT compute_ref_homography_core
  : public vital::algo::compute_ref_homography
{
public:
  PLUGIN_INFO( "core",
               "Default online sequential-frame reference homography estimator." )

  /// Default Constructor
  compute_ref_homography_core();

  /// Default Destructor
  virtual ~compute_ref_homography_core();

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
  virtual void set_configuration( vital::config_block_sptr config );

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
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Estimate the transformation which maps some frame to a reference frame
  /**
   * Similarly to track_features, this class was designed to be called in
   * an online fashion for each sequential frame.
   *
   * \param frame_number frame identifier for the current frame
   * \param tracks the set of all tracked features from the image
   * \return estimated homography
   */
  virtual vital::f2f_homography_sptr
  estimate( vital::frame_id_t frame_number,
            vital::feature_track_set_sptr tracks ) const;

private:

  /// Class storing internal variables
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
