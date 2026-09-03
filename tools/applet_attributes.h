/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_APPLET_ATTRIBUTES_H
#define VIAME_TOOLS_APPLET_ATTRIBUTES_H

namespace viame {
namespace tools {

/// Set on applets that need no plugins loaded on their behalf, letting the
/// tool runner skip the global plugin load. That load pulls in the python
/// module loader, and with it every module named by SPROKIT_PYTHON_MODULES,
/// which costs many seconds.
constexpr char const* SKIP_PLUGIN_PRELOAD = "viame-skip-plugin-preload";

/// Set on applets that parse their own arguments, so that
/// "viame help <applet>" runs the applet with --help instead of printing an
/// empty option list.
constexpr char const* FORWARDS_HELP = "viame-forwards-help";

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_APPLET_ATTRIBUTES_H
