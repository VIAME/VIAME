/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_PYTHON_SCRIPT_APPLET_H
#define VIAME_TOOLS_PYTHON_SCRIPT_APPLET_H

#include <vital/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

#include <string>

namespace viame {
namespace tools {

// ----------------------------------------------------------------------------
/// Runs one of the installed python tools as an applet.
///
/// add_command_options() is deliberately left unimplemented. The base class
/// then tells the tool runner to skip its own argument parsing, so every
/// argument reaches the script untouched and the script owns its command line.
class VIAME_TOOLS_APPLETS_EXPORT python_script_applet
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

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_PYTHON_SCRIPT_APPLET_H
