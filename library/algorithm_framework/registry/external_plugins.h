/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The one door left open to code VIAME was not built with.
///
/// Everything VIAME ships registers by being linked in; see
/// `static_registry.h`. A plugin written outside the tree cannot do that, so
/// `VIAME_PLUGIN_PATH` names the shared libraries to load, and each is asked
/// for `viame_register_plugin`.
///
/// It names **files**, not directories. The loader this replaced listed a
/// directory and opened everything in it, which meant a stale build artifact
/// beside a plugin was loaded as one, and which of two libraries defining the
/// same factory won depended on the order the directory happened to be read
/// in. An explicit list says what is being asked for.

#ifndef VIAME_ALGORITHM_FRAMEWORK_EXTERNAL_PLUGINS_H
#define VIAME_ALGORITHM_FRAMEWORK_EXTERNAL_PLUGINS_H

#include <viame/algorithm_framework/registry/viame_registry_export.h>

#include <string>
#include <vector>

namespace viame {

class registry;

} // namespace viame

namespace viame {

/// The name of the environment variable holding the list.
VIAME_REGISTRY_EXPORT
extern char const* const plugin_path_variable;

/// The variable that named plugin *directories* before P8-T03 took the scan
/// out. It loads nothing now; `register_external_plugins` says so, once, if
/// an environment written for an older VIAME still sets it.
VIAME_REGISTRY_EXPORT
extern char const* const old_plugin_path_variable;

/// The entry point an out-of-tree plugin has to export.
///
/// `extern "C" void viame_register_plugin( viame::registry& )`.
VIAME_REGISTRY_EXPORT
extern char const* const plugin_entry_point;

/// Register the plugins named by `VIAME_PLUGIN_PATH`.
///
/// Does nothing when the variable is unset or empty. A named library that
/// cannot be opened, or that does not export the entry point, is logged and
/// skipped -- one bad entry does not cost the caller the others. A set
/// `KWIVER_PLUGIN_PATH` is warned about on the first call and otherwise
/// ignored.
///
/// @return The libraries whose registration function was called.
VIAME_REGISTRY_EXPORT
std::vector< std::string > register_external_plugins(
  viame::registry& loader );

} // namespace viame

#endif
