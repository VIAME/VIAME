// This file is part of VIAME, and is distributed under an OSI-approved #
// BSD 3-Clause License. See either the root top-level LICENSE file or  #
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

#ifndef VIAME_CORE_TYPES_PYTHON_FOLD_H
#define VIAME_CORE_TYPES_PYTHON_FOLD_H

#include <pybind11/pybind11.h>

/**
 * \file
 * \brief One extension module for `viame.types`, or fifty-six.
 *
 * Each `*_python.cxx` here used to be its own extension module, and fifty-six
 * modules cost 68.6 MB because every one of them carries its own copy of the
 * same instantiated pybind11 and STL templates. Linked as a single module the
 * same code is 16.6 MB, 8.3 MB stripped -- 83% of the exported symbols were
 * duplicates. Measured by linking the existing object files together before
 * any of this was written, so the number is the linker's rather than an
 * estimate.
 *
 * `VIAME_PYTHON_MODULE` is what the translation units say instead of
 * `PYBIND11_MODULE`. Folded, it declares a registration function that the
 * generated master module calls against a submodule of the same name;
 * unfolded, it is `PYBIND11_MODULE` and each file is its own module again.
 * Both spellings build, so the fold is a switch rather than a fork, and a
 * bisect across it does not have to rewrite fifty-six files.
 *
 * **Registration order matters here**, which is the one thing that is not
 * mechanical about the change. `detected_object` references
 * `detected_object_type`, and `homography` references `transform_2d`; as
 * separate modules the order was settled by the `import` lines in
 * `types_init.py`, and `pybind11` raised a clear error when it was wrong.
 * Folded, every submodule registers during one import, so the master calls
 * them in the order `types_init.py` imports them. That file is therefore
 * the definition of the order, and `python.cmake` reads it rather than
 * keeping a second list that could disagree with it.
 */

#ifdef VIAME_PYTHON_FOLD

#define VIAME_PYTHON_MODULE( name, m ) \
  void viame_register_python_##name( ::pybind11::module& m )

/**
 * \brief Declare that this module's types need another module's.
 *
 * As separate modules this was `py::module::import( "viame.types.x" )`, which
 * pulled the dependency in on demand and made registration order irrelevant.
 * Folded it cannot stay: `viame.types.x` is now a shim that imports `_types`,
 * so a registration running during `_types`'s own initialisation re-enters
 * the module being initialised, `PyInit__types` runs a second time, and the
 * first duplicate registration fails with
 *
 *     cannot initialize type "Image": an object with that name is already
 *     defined
 *
 * -- a message that names the first type registered rather than anything to
 * do with the module that caused it, which is what made this worth a comment.
 *
 * Folded, the requirement is met by registering in dependency order instead,
 * so this is a no-op that exists to keep the declaration in the source where
 * `generate_types_fold.py` can read it. The dependencies are *not* derivable
 * any other way, and `types_init.py`'s order does not satisfy them: seventeen
 * of them are back edges there.
 */
#define VIAME_PYTHON_REQUIRE( path ) ( ( void ) 0 )

#else

#define VIAME_PYTHON_MODULE( name, m ) PYBIND11_MODULE( name, m )
#define VIAME_PYTHON_REQUIRE( path ) ::pybind11::module::import( path )

#endif

#endif
