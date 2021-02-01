// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the handle_descriptor_request_core algorithm
 */

#ifndef ARROWS_PLUGINS_CORE_FORMULATE_QUERY_CORE_H_
#define ARROWS_PLUGINS_CORE_FORMULATE_QUERY_CORE_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/algorithm.h>
#include <vital/algo/handle_descriptor_request.h>

#include <vital/algo/image_io.h>
#include <vital/algo/compute_track_descriptors.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A basic query formulator
class KWIVER_ALGO_CORE_EXPORT handle_descriptor_request_core
  : public vital::algo::handle_descriptor_request
{
public:
  PLUGIN_INFO( "core",
               "Formulate descriptors for later queries." )

  /// Default Constructor
  handle_descriptor_request_core();

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

  /// Formulate query
  virtual bool handle(
    kwiver::vital::descriptor_request_sptr request,
    kwiver::vital::track_descriptor_set_sptr& desc,
    std::vector< kwiver::vital::image_container_sptr >& imgs );

private:

  /// The feature detector algorithm to use
  vital::algo::image_io_sptr reader_;

  /// The descriptor extractor algorithm to use
  vital::algo::compute_track_descriptors_sptr extractor_;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
