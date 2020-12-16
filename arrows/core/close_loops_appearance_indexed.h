// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Appearance indexed close loops algorithm impl interface
 */

#ifndef KWIVER_CORE_CLOSE_LOOPS_APPEARANCE_INDEXED_H_
#define KWIVER_CORE_CLOSE_LOOPS_APPEARANCE_INDEXED_H_

#include <string>

#include <vital/vital_config.h>
#include <vital/algo/close_loops.h>

#include <arrows/core/kwiver_algo_core_export.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Loop closure algorithm that using appearance indexing for fast matching
class KWIVER_ALGO_CORE_EXPORT close_loops_appearance_indexed
  : public kwiver::vital::algo::close_loops
{
public:

  PLUGIN_INFO( "appearance_indexed",
               "Uses bag of words index to close loops." )

  /// Default constructor
  close_loops_appearance_indexed();

  /// Find loops in a feature_track_set

  /// Attempt to perform closure operation and stitch tracks together.
  /**
  * \param frame_number the frame number of the current frame
  * \param input the input feature track set to stitch
  * \param image image data for the current frame
  * \param mask Optional mask image where positive values indicate
  *                  regions to consider in the input image.
  * \returns an updated set of feature tracks after the stitching operation
  */
  virtual kwiver::vital::feature_track_set_sptr
    stitch(kwiver::vital::frame_id_t frame_number,
      kwiver::vital::feature_track_set_sptr input,
      kwiver::vital::image_container_sptr image,
      kwiver::vital::image_container_sptr mask = kwiver::vital::image_container_sptr()) const;

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

protected:
  /// the feature m_detector algorithm
  class priv;
  std::shared_ptr<priv> d_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
