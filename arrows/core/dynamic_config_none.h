// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the interface to dynamic_config_none
 */

#ifndef ARROWS_CORE_DYNAMIC_CONFIG_NONE_H
#define ARROWS_CORE_DYNAMIC_CONFIG_NONE_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/dynamic_configuration.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A class for bypassing image conversion
class KWIVER_ALGO_CORE_EXPORT dynamic_config_none
  : public vital::algo::dynamic_configuration
{
public:
  PLUGIN_INFO( "none",
               "Null implementation of dynamic_configuration.\n\n"
               "This algorithm always returns an empty configuration block." )

  /// Default constructor
  dynamic_config_none();

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Return dynamic configuration values
  /**
   * This method returns dynamic configuration values. A valid config
   * block is returned even if there are not values being returned.
   */
  virtual kwiver::vital::config_block_sptr get_dynamic_configuration();
};

} } } // end namespace

#endif /* ARROWS_CORE_DYNAMIC_CONFIG_NONE_H */
