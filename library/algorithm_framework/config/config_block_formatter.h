// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_CONFIG_FORMATTER_H
#define KWIVER_CONFIG_FORMATTER_H

#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/config/viame_config_export.h>

#include <ostream>
#include <string>

namespace viame {

/// @brief Generates formatted versions of a config block.
///
/// P8-T09 removed three of its four entry points. `set_prefix` and
/// `generate_source_loc` had no caller, so the prefix was always empty and
/// the source location always printed -- the options existed but only ever
/// held their defaults, and `print` branched on constants. `format_block`
/// was worse: declared private, never defined, so any caller would have got
/// a link error rather than an answer.
///
/// This class encapsulates several different formatting options for
/// a config block.
///
/// TODO: This likely should be an "algorithm" and not be located in the config
///       module since there is no inbuilt use of this -- It seems to only be
///       used in down-stream libraries/tools.
class VIAME_CONFIG_EXPORT config_block_formatter
{
public:
  config_block_formatter( const config_block_sptr config );
  ~config_block_formatter() = default;

  /// @brief Format config block in simple text format.
  ///
  /// One key per line, sorted, with `[RO]` on a read-only key and the file
  /// and line a value came from when it came from one.
  ///
  /// @param str Stream to format on.
  void print( std::ostream& str );

private:
  config_block_sptr m_config;
};

} // namespace viame

#endif // KWIVER_CONFIG_FORMATTER_H
