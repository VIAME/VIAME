/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Registration without a plugin loader.
///
/// Every VIAME library that provides factories has a registration function --
/// `register_factories`, the entry point kwiver's loader used to `dlopen` a
/// module to find. P8-T03 links those libraries normally and calls the
/// functions directly, so there is no module to find and no directory to
/// search.
///
/// Each library's entry point is renamed at compile time rather than in its
/// source: the registration file is compiled into the library with
/// `-Dregister_factories=viame_register_<library>`, which gives it a unique
/// symbol without editing any of the files that define one. The generated
/// `static_registry.cxx` declares them and calls them in turn.
///
/// A library that is not built contributes nothing, because CMake only adds
/// it to the generated list when `viame_register_statically` ran for it.

#ifndef VIAME_ALGORITHM_FRAMEWORK_STATIC_REGISTRY_H
#define VIAME_ALGORITHM_FRAMEWORK_STATIC_REGISTRY_H

#include <viame/algorithm_framework/registry/viame_registry_export.h>

#include <viame/algorithm_framework/plugin/plugin_manager.h>

namespace viame {

/// Register the factories VIAME was built with.
///
/// `types` selects which, by the same bits `load_all_plugins` takes: a
/// library answers to the one its plugin used to be installed under, so
/// asking for `APPLETS` still costs nothing but the applets.
///
/// Idempotent: each library's registration guards on its own module name, so
/// a second call registers nothing.
VIAME_REGISTRY_EXPORT
void register_builtins(
  kwiver::vital::plugin_loader& loader,
  kwiver::vital::plugin_manager::plugin_types types );

} // namespace viame

#endif
