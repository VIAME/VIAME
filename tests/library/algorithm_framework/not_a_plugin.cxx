/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief A real shared library that is not a plugin.
///
/// The case `a_library_without_the_entry_point_is_skipped` is about: a file
/// the loader can open and find no `viame_register_plugin` in.
///
/// It used to point at **libviame itself**, and that was a trap rather than
/// an economy. The test binary links libviame, and under `setup_viame.sh` it
/// links the *install* copy while the test `dlopen`s the *build* copy by
/// path -- so the process ended up with two distinct loads of the same
/// library, two sets of its globals, and a heap that corrupted on the way
/// out: the test passed and then the process died with
/// `malloc_consolidate(): invalid chunk size`. The loader never closes a
/// handle, deliberately, so nothing undid it.
///
/// This carries a symbol so that it is not an empty object, and no entry
/// point, which is the whole of what the test needs.

#ifdef _WIN32
#define PLUGIN_EXPORT_FLAG __declspec( dllexport )
#else
#define PLUGIN_EXPORT_FLAG __attribute__( ( visibility( "default" ) ) )
#endif

extern "C"
PLUGIN_EXPORT_FLAG
int
viame_test_not_a_plugin_marker( void )
{
  return 1;
}
