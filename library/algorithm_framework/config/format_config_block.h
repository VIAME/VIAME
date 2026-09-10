// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file This header defines the interface for a plugin that formats a
/// config block.

#ifndef VITAL_CONFIG_FORMAT_CONFIG_BLOCK_H
#define VITAL_CONFIG_FORMAT_CONFIG_BLOCK_H

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/pluggable.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

#include <ostream>
#include <string>

namespace kwiver::vital {

/// Config block printer plugin interface.
// ----------------------------------------------------------------------------
/// This class defines the abstract interface for all implementations
/// of the config block formatting plugin.
///
/// TODO: This likely should be an "algorithm" and not be located in the config
///       module since there is no inbuilt use of this -- It seems to only be
///       used in down-stream libraries/tools.
class format_config_block : public pluggable
{
public:
  PLUGGABLE_INTERFACE( format_config_block )

  virtual void print( const config_block_sptr config, std::ostream& str ) = 0;

  virtual void set_configuration( [[maybe_unused]] config_block_sptr ) {}

  virtual void
  set_configuration_internal( [[maybe_unused]] config_block_sptr )
  {}

  virtual config_block_sptr
  get_configuration() const { return nullptr; }

protected:
  virtual void initialize() {}
}; // end class format_config_block

using format_config_block_sptr = std::shared_ptr< format_config_block >;

} // end namespace

#endif // VITAL_CONFIG_FORMAT_CONFIG_BLOCK_H
