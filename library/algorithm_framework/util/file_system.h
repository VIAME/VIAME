// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief The file system operations VIAME uses.
///
/// Two hundred call sites across the tree asked `kwiversys::SystemTools` for
/// these. P8-T05 is replacing kwiversys with the standard library, and this
/// is the seam: the call sites name these functions, and what is underneath
/// them changes once rather than two hundred times.
///
/// The semantics are **kwiversys's**, not `std::filesystem`'s, and they are
/// recorded in `library/algorithm_framework/tests/test_file_system.cxx`. They
/// differ in at least four places -- see that file -- and a call site
/// rewritten directly against `std::filesystem` would have changed behaviour
/// in ways nothing would have caught.

#ifndef KWIVER_VITAL_UTIL_FILE_SYSTEM_H
#define KWIVER_VITAL_UTIL_FILE_SYSTEM_H

#include <viame/algorithm_framework/util/vital_util_export.h>

#include <string>
#include <vector>

namespace viame {

// Asking about a path ---------------------------------------------------------

/// @brief Is there anything at this path?
///
/// True for a directory as well as a file. An empty path is false rather than
/// an error.
VITAL_UTIL_EXPORT bool file_exists( std::string const& path );

/// @brief Is there a directory at this path?
///
/// A trailing separator makes no difference.
VITAL_UTIL_EXPORT bool file_is_directory( std::string const& path );

/// @brief Is there a file -- not a directory -- at this path?
///
/// What `FileExists( path, true )` asked, which three call sites wanted: a
/// directory of the right name is not an answer to "is the model there?".
VITAL_UTIL_EXPORT bool file_is_regular( std::string const& path );

/// @brief Is this path absolute?
VITAL_UTIL_EXPORT bool file_is_full_path( std::string const& path );

// Taking a path apart ---------------------------------------------------------

/// @brief Everything before the last component, without a trailing separator.
///
/// `"/a/b/c.txt"` gives `"/a/b"`, and so does `"/a/b/c.txt/"`: the trailing
/// separator is dropped before the split, which is where this parts company
/// with `std::filesystem::path::parent_path`.
VITAL_UTIL_EXPORT std::string filename_path( std::string const& path );

/// @brief The last component.
///
/// Empty when the path ends in a separator.
VITAL_UTIL_EXPORT std::string filename_name( std::string const& path );

/// @brief The last extension, including its dot, or empty.
///
/// `"a.tar.gz"` gives `".gz"`. A name that is nothing but an extension --
/// `".bashrc"` -- gives the whole of it, which `std::filesystem` does not.
VITAL_UTIL_EXPORT std::string filename_last_extension(
  std::string const& path );

/// @brief The last component without its last extension.
///
/// `".bashrc"` gives the empty string; see `filename_last_extension`.
VITAL_UTIL_EXPORT std::string filename_without_last_extension(
  std::string const& path );

/// @brief The directory this path is in.
VITAL_UTIL_EXPORT std::string parent_directory( std::string const& path );

/// @brief Split a path into its root and components.
///
/// The first element is the root -- `"/"` for an absolute path, empty for a
/// relative one -- and the rest are the components. `join_path` is the
/// inverse.
VITAL_UTIL_EXPORT void split_path(
  std::string const& path, std::vector< std::string >& components );

/// @brief Put back together what `split_path` took apart.
VITAL_UTIL_EXPORT std::string join_path(
  std::vector< std::string > const& components );

// Making a path absolute ------------------------------------------------------

/// @brief Absolute, with `.` and `..` resolved textually.
///
/// Relative to `base` if given, to the working directory otherwise. The path
/// need not exist. A trailing separator is dropped.
VITAL_UTIL_EXPORT std::string collapse_full_path( std::string const& path );

VITAL_UTIL_EXPORT std::string collapse_full_path(
  std::string const& path, std::string const& base );

/// @brief Absolute, with symbolic links resolved.
VITAL_UTIL_EXPORT std::string real_path( std::string const& path );

/// @brief The working directory.
VITAL_UTIL_EXPORT std::string current_working_directory();

/// @brief Turn every backslash into a forward slash.
///
/// Also drops a trailing separator.
VITAL_UTIL_EXPORT void convert_to_unix_slashes( std::string& path );

// Changing the file system ----------------------------------------------------

/// @brief Make this directory and any parents it needs.
///
/// @return Whether it exists afterwards.
VITAL_UTIL_EXPORT bool make_directory( std::string const& path );

/// @brief Remove a directory and everything in it.
VITAL_UTIL_EXPORT bool remove_directory( std::string const& path );

/// @brief Remove one file.
VITAL_UTIL_EXPORT bool remove_file( std::string const& path );

// Looking things up -----------------------------------------------------------

/// @brief Find a named file in a list of directories.
///
/// @return The path found, or empty.
VITAL_UTIL_EXPORT std::string find_file(
  std::string const& name, std::vector< std::string > const& directories );

/// @brief Find an executable on PATH, or in the directories given.
VITAL_UTIL_EXPORT std::string find_program(
  std::string const& name,
  std::vector< std::string > const& directories = {} );

/// @brief The names in a directory, including `.` and `..`.
///
/// Empty if the path is not a directory. The order is the file system's.
VITAL_UTIL_EXPORT std::vector< std::string > directory_entries(
  std::string const& path );

/// @brief Can this process read this file?
///
/// Not the same question as `file_exists`: a file can be there and not be
/// readable, and the pipeline parser asks this one so that it can say which
/// of the two went wrong.
VITAL_UTIL_EXPORT bool file_is_readable( std::string const& path );

// The environment -------------------------------------------------------------

/// @brief Read an environment variable.
///
/// @return Whether it was set. `value` is untouched when it was not.
VITAL_UTIL_EXPORT bool get_env( std::string const& name, std::string& value );

/// @brief Read an environment variable.
///
/// @return Its value, or nullptr.
VITAL_UTIL_EXPORT char const* get_env( std::string const& name );

/// @brief Append the directories named by a PATH-style variable.
///
/// Separated by `:` on Unix and `;` on Windows. Existing contents are kept
/// and the directories are appended, so the caller's own entries come first.
/// An empty entry -- what `a::b` has in the middle -- is appended as an empty
/// string rather than dropped, because dropping it would change which
/// directory a search found things in.
///
/// Nothing is appended when the variable is not set.
VITAL_UTIL_EXPORT void environment_path(
  std::string const& name, std::vector< std::string >& directories );

} // namespace viame

#endif // KWIVER_VITAL_UTIL_FILE_SYSTEM_H
