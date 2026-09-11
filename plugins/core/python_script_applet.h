/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_PYTHON_SCRIPT_APPLET_H
#define VIAME_CORE_PYTHON_SCRIPT_APPLET_H

#include <viame/algorithm_framework/applets/kwiver_applet.h>

#include "viame_core_export.h"

#include <string>

#include <vector>

namespace viame {

// ----------------------------------------------------------------------------
/// Locate an installed tool script (configs/<name>), or return an empty string.
VIAME_CORE_EXPORT std::string find_tool_script( const std::string& name );

/// Run an installed tool script with the given arguments, sharing this
/// process's streams, and return its exit code.
VIAME_CORE_EXPORT int run_tool_script( const std::string& script,
                                       const std::vector< std::string >& args );

// ----------------------------------------------------------------------------
/// Runs one of the installed python tools as an applet.
///
/// add_command_options() is deliberately left unimplemented. The base class
/// then tells the tool runner to skip its own argument parsing, so every
/// argument reaches the script untouched and the script owns its command line.
class VIAME_CORE_EXPORT python_script_applet
  : public kwiver::tools::kwiver_applet
{
public:
  int run() override;

protected:
  /// Script file name, as installed into the configs directory.
  virtual std::string script_name() const = 0;
};

// ----------------------------------------------------------------------------
/// Declare an applet that forwards to the named python tool.
#define VIAME_PYTHON_SCRIPT_APPLET( cls, applet_name, script, description ) \
class cls : public python_script_applet                                     \
{                                                                           \
public:                                                                     \
  PLUGIN_INFO( applet_name, description )                                   \
                                                                            \
protected:                                                                  \
  std::string script_name() const override { return script; }               \
};

} // namespace viame

#endif // VIAME_CORE_PYTHON_SCRIPT_APPLET_H
