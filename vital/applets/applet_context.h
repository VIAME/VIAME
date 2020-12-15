// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOLS_APP_CONTEXT_H
#define KWIVER_TOOLS_APP_CONTEXT_H

#include <vital/util/wrap_text_block.h>

#include <vital/applets/cxxopts.hpp>

#include <memory>
#include <ostream>

namespace kwiver {
namespace tools {

// ----------------------------------------------------------------
/**
 * @brief Applet context provided by the tool runner.
 *
 * This class contains data that are shared between the tool runner
 * and the applet.
 */
class applet_context
{
public:
  // Used to wrap large text blocks
  kwiver::vital::wrap_text_block m_wtb;

  // name of the applet. as in kwiver <applet> <args..>
  std::string m_applet_name;

    /**
   * Results from parsing the command options. Note that you do not
   * own this storage.
   */
  cxxopts::ParseResult*  m_result { nullptr };

  // Original args for plugin for alternate command line processing.
  std::vector< std::string >m_argv;

  // Flag for skipping command line parsing
  bool m_skip_command_args_parsing { false };

}; // end class applet_context

} } // end namespace

#endif /* KWIVER_TOOLS_APP_CONTEXT_H */
