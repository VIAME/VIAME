// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file This header defines the interface for a plugin that formats a
 * config block.
 */

#ifndef VITAL_CONFIG_FORMAT_CONFIG_BLOCK_H
#define VITAL_CONFIG_FORMAT_CONFIG_BLOCK_H

#include <vital/config/config_block.h>

#include <ostream>
#include <string>

namespace kwiver {
namespace vital {

/// Config block printer plugin interface.
// ----------------------------------------------------------------
/**
 * This class defines the abstract interface for all implementations
 * of the config block formatting plugin.
 *
 */
class format_config_block
{
public:
  // -- CONSTRUCTORS --
  virtual ~format_config_block() = default;

  virtual void print( std::ostream& str ) = 0;

  // Options that are passed from the main calling context

  bool opt_gen_source_loc;
  std::string opt_prefix;

  // The config block to format.
  config_block_sptr m_config;

protected:
  format_config_block() { }

}; // end class format_config_block

using format_config_block_sptr = std::shared_ptr< format_config_block >;

} } // end namespace

#endif /* VITAL_CONFIG_FORMAT_CONFIG_BLOCK_H */
